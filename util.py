import argparse
import pandas as pd
import json
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from scipy import stats
import ast


def create_merged_df(pickle_file_path, annotations_file_path):
    """
        instances.json has 208960 elements
        refs(umd).p    has 49822  elements
        we return the merged dataframe that has the same elements as the ref file (in the case of the umd one 49822)
    """
    

    with open(annotations_file_path, 'r') as f:
        annotations = json.load(f)["annotations"]
        annotations = pd.DataFrame(annotations)

    # we remove also the image_id since they will be merged on the bbox id which is more specific
    columns_to_remove = ["segmentation", "area", "iscrowd", "category_id", "image_id"] # possible: segmentation, area, iscrowd, image_id, bbox, category_id, id
    annotations = annotations.drop(columns=columns_to_remove)


    partition = pd.read_pickle(pickle_file_path) # read Pickle file
    partition = pd.DataFrame.from_records(partition) # convert it to a dataframe

    
    columns_to_remove = ["category_id", "sent_ids", "ref_id", "image_id"] # possible: image_id, split, sentences, file_name, category_id, ann_id, sent_ids, ref_id
    partition = partition.drop(columns=columns_to_remove)


    # We merge on id == ann_id which are the descriptions of the bboxes
    merged = pd.merge(annotations, partition, left_on='id', right_on='ann_id', how='inner')

    # we decide, for now, to use the first sentence/description of the box. Some bboxes have several descriptions for the same bbox
    #print(merged.head()['sentences'])
    merged['sentences'] = merged['sentences'].apply(lambda x: x[0]['sent'])
    #print(merged.head()['sentences'])


    #now we refine the file name by adding the root dir and by cleaning the name (in particular it has an additional number before the extension which the images does not have)

    merged['file_name'] = merged['file_name'].apply(modify_filename)

    # we don't need the id anymore
    columns_to_remove = ["ann_id", "id"]
    merged = merged.drop(columns=columns_to_remove)

    # merged has ['bbox', 'split', 'sentences', 'filename'], everything we need and nothing more. Note that we have 3 splits, train, val and test.
    return merged


def modify_filename(file_name):
    """
        function for modifying the file_name of each image in the dataframe
    """

    # Remove the last number before the extension
    base_name = '_'.join(file_name.split('_')[:-1]) + '.jpg'
    
    # Add the root dictory of the dataset
    new_name = f'/home/rtx/deep_learning/dataset/refcocog/images/{base_name}'
    
    
    return new_name


class VisualGroundingRefcocog(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, tokenizer, prefix_description=None, modified_clip_preprocess=None, bbox_transform=None):
        
        self.images = dataset['file_name'].tolist()
        self.descriptions = dataset['sentences'].tolist()
        #self.bboxes = [xywh2xyxy(ast.literal_eval(bbox)) for bbox in dataset['bbox'].tolist()]
        self.bboxes = [xywh2xyxy(bbox) for bbox in dataset['bbox'].tolist()]
        self.transform = modified_clip_preprocess
        self.bbox_transform = bbox_transform
        self.tokenizer = tokenizer
        self.prefix_description = prefix_description


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        image = Image.open(self.images[idx])#.convert("RGB")
        description = self.descriptions[idx]
        bbox = self.bboxes[idx]

        # We directly apply the CLIP's modified preprocess to the images
        """
            Why we modify the CLIP's preprocess? Because it performs a center crop of the image
            and we dont want that. We want to keep the whole image and not risk to cut away our
            object of interest.
        """

        if self.prefix_description is not None:
            description = self.prefix_description + description
        description = self.tokenizer(description).squeeze() # (1, 77) -> (77)

        if self.transform:
            original_width, original_height = image.size
            image = self.transform(image)
        
        bbox_ratio = calculate_bbox_ratio(original_width, original_height, bbox)

        if self.bbox_transform:
            bbox = self.bbox_transform(bbox, (original_width, original_height), (224, 224))


        # For AttBalance
        bbox_mask = create_bbox_mask((7, 7), bbox)

        sample = {
            'image': image,
            'description': description,
            'bbox': bbox,
            'bbox_mask': bbox_mask,
            'bbox_ratio': bbox_ratio,
        }

        return sample


def get_dataloader(dataset, batch_size):

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=2
    )

    return data_loader


def calculate_IoU(x_bbox, y_bbox):
    """
    Calculate IoU between two batches of bounding boxes.
    x_bbox, y_bbox: tensors of shape (batch_size, 4)
    Returns: tensor of IoU values, shape (batch_size)
    """

    # get intersection's coordinates
    inter_xmin = torch.max(x_bbox[:, 0], y_bbox[:, 0])
    inter_ymin = torch.max(x_bbox[:, 1], y_bbox[:, 1])
    inter_xmax = torch.min(x_bbox[:, 2], y_bbox[:, 2])
    inter_ymax = torch.min(x_bbox[:, 3], y_bbox[:, 3])

    # clamp to zero when no intersection
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    intersection = inter_w * inter_h

    # areas
    area_x = (x_bbox[:, 2] - x_bbox[:, 0]) * (x_bbox[:, 3] - x_bbox[:, 1])
    area_y = (y_bbox[:, 2] - y_bbox[:, 0]) * (y_bbox[:, 3] - y_bbox[:, 1])
    union = area_x + area_y - intersection

    iou = intersection / union.clamp(min=1e-6)

    return iou


def criterion_iou(x_bbox, target_bbox, l1_weight=1.0, ciou_weight=1.0):
    """
    Combines Smooth L1 loss e CIoU loss for bbox coordinates regression.
    l1_weight: smooth_l1 weigh
    giou_weight: CIoU weight
    """
    # Both sets of boxes are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2
    # and The two boxes should have the same dimensions.
    ciou = torchvision.ops.complete_box_iou_loss(x_bbox, target_bbox, reduction='mean')

    l1 = nn.functional.smooth_l1_loss(x_bbox, target_bbox)
    #print(f"SmoothL1: {l1_weight * l1.item():.2f}, CIoU: {giou_weight * ciou.item():.2f}, Total: {(l1 + ciou).item():.2f}")
    return l1_weight * l1 + ciou_weight * ciou


def only_ciou(x_bbox, target_bbox):
    ciou = torchvision.ops.complete_box_iou_loss(x_bbox, target_bbox, reduction='mean')
    return ciou


def xywh2xyxy(bbox):
    '''
    refcocog labels are in the format x y w h
    yolo are xmin ymin xmax ymax

    Input format:
            bbox         = [x, y, w, h]
        Output:
            updated_bbox = [x, y, x, y]
    '''
    xmin, ymin, w, h = bbox

    xmax = xmin + w
    ymax = ymin + h

    updated_bbox = [xmin, ymin, xmax, ymax]

    return updated_bbox


def cxcywh_to_xyxy(bboxes):
    """
    Converts a batch of bboxes from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)
    """
    # faster than using x_bbox[:, 0], y_bbox[:, 0]
    cx, cy, w, h = bboxes.unbind(-1)
    
    w_half = w / 2.0
    h_half = h / 2.0
    
    x1 = cx - w_half
    y1 = cy - h_half
    x2 = cx + w_half
    y2 = cy + h_half

    return torch.stack([x1, y1, x2, y2], dim=-1)




def draw_bbox(image, bbox, color, save_path=None, caption=None):
    """
    image is a tensor (C, H, W) and bbox is an array of the 4 coordinates
    """
    xmin, ymin, xmax, ymax = bbox

    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)

    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
    if caption is not None:
        draw.text((5, 5), caption, fill="white")
    
    if save_path is not None:
        image.save(save_path)
    else:
        image.show()



def compare_bbox(image, pred_bbox, label_bbox, save_path=None, caption=None, color1="green", color2="red"):
    #img = Image.open(image_path)
    
    xmin1, ymin1, xmax1, ymax1 = label_bbox
    xmin2, ymin2, xmax2, ymax2 = pred_bbox

    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Create rectangle patches
    rect1 = patches.Rectangle((xmin1, ymin1),
                             xmax1-xmin1,
                             ymax1-ymin1,
                             linewidth=2,
                             edgecolor=color1,
                             facecolor='none')
    # Predicted one
    rect2 = patches.Rectangle((xmin2, ymin2),
                             xmax2-xmin2,
                             ymax2-ymin2,
                             linewidth=2,
                             edgecolor=color2,
                             facecolor='none')
    
    # Add rectangles to axes
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    # Add description
    if caption is not None:
        plt.suptitle(caption, fontsize=8)
    ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    

def compare_bbox_old(image, pred_bbox, label_bbox, save_path=None, caption=None, color1="green", color2="red"):
    #img = Image.open(image_path)
    
    xmin1, ymin1, xmax1, ymax1 = label_bbox
    xmin2, ymin2, xmax2, ymax2 = pred_bbox

    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)
    
    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin1, ymin1, xmax1, ymax1], outline=color1, width=3)
    draw.rectangle([xmin2, ymin2, xmax2, ymax2], outline=color2, width=3)
    if caption is not None:
        draw.text((1, 1), caption, fill="white", stroke_width=0.1)
    
    if save_path is not None:
        image.save(save_path)
    else:
        image.show()


def modified_clip_preprocess(keep_aspect_ratio):
    """
    The original CLIP's preprocess is the following:
    Compose(
        Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
        CenterCrop(size=(224, 224))
        <function _convert_image_to_rgb at 0x744263e27ba0>
        ToTensor()
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    )
    """
    """
    Note that torchvision.transforms.Compose expects callables (i.e., functions or objects you can "call") as transformations
    
    transforms.Lambda(lambda img: img.convert("RGB")) is equivalent to
    
    def convert_rgb(img):
        return img.convert("RGB")

    transforms.Lambda(convert_rgb)
    """

    def resize_pad(img):
        # Resize mantaining the same aspect ratio (longest dim = 224)
        width, height = img.size
        if width >= height:
            new_width = 224
            new_height = int(224 * height / width)
        else:
            new_height = 224
            new_width = int(224 * width / height)
        
        resized_img = img.resize((new_width, new_height), resample=Image.BICUBIC)
        
        # Centered Padding to obtain 224x224
        pad_width = 224 - new_width
        pad_height = 224 - new_height
        padding = (
            pad_width // 2, 
            pad_height // 2, 
            pad_width - (pad_width // 2), 
            pad_height - (pad_height // 2)
        )
        return transforms.functional.pad(resized_img, padding, fill=0, padding_mode="constant")


    transform_steps = []
    
    if keep_aspect_ratio:
        transform_steps.append(transforms.Lambda(resize_pad))
    else:
        transform_steps.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC, antialias=True)) #-> this stretches the image
 
    transform_steps.extend([
        transforms.Lambda(lambda img: img.convert("RGB")), # to rgb
        transforms.ToTensor(), # from PIL image to tensor
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    
    return transforms.Compose(transform_steps)


def resize_bbox(bbox, original_size, new_size, keep_aspect_ratio):
    original_width, original_height = original_size
    new_width, new_height = new_size
    norm = max(new_height, new_width) # in our case, wehere new size is 224, 224 will be 224

    if keep_aspect_ratio:
        if original_width >= original_height:
            resized_width = 224
            resized_height = int(224 * original_height / original_width)
        else:
            resized_height = 224
            resized_width = int(224 * original_width / original_height)
        scale_x = resized_width / original_width
        scale_y = resized_height / original_height
        
        # padding
        pad_left = (224 - resized_width) // 2
        pad_top = (224 - resized_height) // 2
    else:
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        # we do not have any padding
        pad_left = 0
        pad_top = 0

    x1, y1, x2, y2 = bbox
    
    resized_bbox = [
        x1 * scale_x + pad_left,
        y1 * scale_y + pad_top,
        x2 * scale_x + pad_left,
        y2 * scale_y + pad_top
    ]
    # To tensor
    resized_bbox = torch.tensor(resized_bbox, dtype=torch.float32)
    # Normalize the bbox between 0 and 1
    resized_bbox = resized_bbox / norm

    return resized_bbox


def create_bbox_mask(dim, bbox):
    """
    Create a 7x7 binary mask with 1s inside the bounding box, then flatten it to have a 49 one.
    """
    h, w = dim

    tmp_bbox = bbox.clone() # if we dont make a copy we'll modify the 'original' bbox, Python does this for tensors and numpy array (mutable object)

    tmp_bbox *= h # we have the bbox normalized (/224) and want to project them into a 7x7 grid

    tmp_bbox = tmp_bbox.numpy()
    x_min, y_min, x_max, y_max = tmp_bbox

    x_min = int(x_min)
    x_max = int(x_max)
    y_min = int(y_min)
    y_max = int(y_max)

    # Create a 7x7 mask filled with zeros 
    mask = torch.zeros((h, w), dtype=torch.float32)
    # Set grid cells to 1 inside bbox
    mask[y_min:y_max, x_min:x_max] = 1
    mask = mask.flatten()
    return mask


def calculate_bbox_ratio(width_image, height_image, bbox):
    
    x1, y1, x2, y2 = bbox

    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = width_image * height_image

    bbox_ratio = bbox_area / image_area

    return bbox_ratio


def denormalize_data(image, bbox1, bbox2, description):
    """
        Inverts the normalization we applied to the data
        for a meaningful visualization.
    """
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    image = image.clone()
    
    # denorm channel by channel
    for t, mean, std in zip(image, CLIP_MEAN, CLIP_STD):
        t.mul_(std).add_(mean)
    
    # Clamp the values between 0 and 1
    image = torch.clamp(image, 0, 1)

    bbox1 = bbox1 * 224
    bbox2 = bbox2 * 224

    tokenizer = SimpleTokenizer()
    description = description.tolist()
    eos_index = description.index(49407)
    description = description[1:eos_index]
    
    decoded_text = tokenizer.decode(description)

    return image, bbox1, bbox2, decoded_text



def init_weights(mat):
    """
    m: module (like "nn.Linear", "nn.LSTM")
    m_name : name of the module (like "visual_projection_layer")
    """
    for m_name, m in mat.named_modules():
        if type(m) in [nn.Linear]:
            if "visual_projection_layer" in m_name or "richer_visual_projection_layer" in m_name or "textual_projection_layer" in m_name:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    #print(f"Resuming from epoch {epoch} with loss {loss}")
    return model, optimizer, epoch, loss


def compute_softmax_mean_heads_attention(attention_all_heads):
    """
        We need to:
                    1. extract the attention between the reg token and the visual ones
                    2. average the attention of all heads
                    3. apply the softmax
    """
    # (batch_size, heads, seq_len, seq_len) -> (batch_size, seq_len, seq_len) ## average among heads
    mean_result = torch.mean(attention_all_heads, dim=1)
    # (batch_size, seq_len, seq_len) -> (batch_size, seq_len, seq_len)
    softmax_result = mean_result.softmax(dim = -1)
    # (batch_size, seq_len, seq_len) -> (batch_size, 49) ## only reg-visual tokens
    reg_visual_attention = softmax_result[:, -1, :49]

    return reg_visual_attention


def compute_relative_rho(rho):

    all_rhos = torch.stack(list(rho.values()))
    # Perform the summation of the rhos divided by num_layers -> mean
    mean_rho_per_example = all_rhos.mean(dim=0)

    # relative_rho = rho - mean + 1
    relative_rho = {
        layer: rho - mean_rho_per_example + 1
        for layer, rho in rho.items()
    }

    return relative_rho


def rac_loss(realtive_rhos, attention_values, masks, eps=1e-6):
    
    inverted_masks = 1 - masks

    all_relative_rhos = torch.stack(list(realtive_rhos.values()))

    all_attention_values = torch.stack(attention_values)

    # Hadamard products (elementwise multiplication) -> (num_layers, batch_size, 49)
    # We broadcast the masks on each layer
    attention_in_mask = all_attention_values * masks.unsqueeze(0)          # * M ## element-wise multiplication
    attention_out_mask = all_attention_values * inverted_masks.unsqueeze(0) # * M-


    # Sum over the 49 spatial positions -> (num_layers, batch_size)
    sum_in = attention_in_mask.sum(dim=-1)
    sum_out = attention_out_mask.sum(dim=-1)

    # Compute the logs (add eps to prevent log(0))
    log_in = torch.log(sum_in + eps)                  # log(summation attention_i * M)
    log_out = torch.log(1 - sum_out + eps)            # log(summation attention_i * M-)

    # Final loss: shape (6, B)
    per_layer_loss = all_relative_rhos * (-log_in) - log_out

    # (6, batch_size) -> (, batch_size)
    per_example_loss = per_layer_loss.sum(dim=0)  # sum over layers 

    rac_loss = per_example_loss.mean()

    return rac_loss


def spearmanr_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Spearman's rank correlation coefficient (ρ) for each pair of vectors in the batch.

    Args:
        x: Tensor of shape (batch_size, dim)
        y: Tensor of shape (batch_size, dim)

    Returns:
        rho: Tensor of shape (batch_size,) with Spearman's rho for each sample in the batch
    """

    def compute_ranks(tensor):
        # Get ranks using double argsort
        sorted_indices = torch.argsort(tensor, dim=1)
        ranks = torch.argsort(sorted_indices, dim=1).float()
        return ranks

    # 1. Get the ranks
    x_rank = compute_ranks(x)
    y_rank = compute_ranks(y)

    # 2. Normalize the ranks (optional for stability)
    x_rank = x_rank - x_rank.mean(dim=1, keepdim=True)
    y_rank = y_rank - y_rank.mean(dim=1, keepdim=True)

    # 3. Compute the Spearman correlation via Pearson on the ranks
    numerator = (x_rank * y_rank).sum(dim=1)
    denominator = torch.sqrt((x_rank ** 2).sum(dim=1) * (y_rank ** 2).sum(dim=1))
    rho = numerator / denominator

    return rho


def mrc_loss(student_attentions, teacher_attentions):
    """
    Computes the KL divergence loss between student and teacher attention maps.
    """
    # Ensure teacher is normalized (probabilities) and student is in log-space
    student_log = torch.log_softmax(student_attentions, dim=-1)  # (6, B, 49)
    teacher_prob = torch.softmax(teacher_attentions, dim=-1)     # (6, B, 49)

    # KL divergence: sum over tokens (dim=-1), then mean over batch, then sum over layers
    kl = nn.functional.kl_div(student_log, teacher_prob, reduction='batchmean')  # scalar
    return kl



@torch.no_grad()
def update_momentum(model, momentum_model, m=0.9):
    for param_q, param_k in zip(model.parameters(), momentum_model.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)