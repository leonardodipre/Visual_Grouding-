import torch
import clip
from baseline import yolo_inference, clip_inference
from util import calculate_IoU, cxcywh_to_xyxy, compare_bbox, draw_bbox, denormalize_data
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_loop_baseline(clip_model, clip_preprocess, yolo_model, data):
    correct = 0
    total = 0
    iou_array = []
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in tqdm(data, desc="Processing Test Data"):
            images = sample['image']
            for i, image in enumerate(images):
                cropped_images, bboxes = yolo_inference(yolo_model, image)
                pred_bbox = torch.tensor(clip_inference(clip_model, clip_preprocess, cropped_images, bboxes, sample['description'][i]), dtype=torch.float32)

                iou = calculate_IoU(pred_bbox.unsqueeze(0), sample['bbox'][i].unsqueeze(0))

                iou_array.append(iou)

                if  iou >= 0.5:
                    correct += 1
                total += 1

    return correct/total, np.asarray(iou_array).mean()


def train_loop(model, data, optimizer, criterion_iou, device):
    model.train()
    loss_array = []
    for sample in tqdm(data, desc="Processing Training Dataset"):
    #for sample in data:
        images = sample["image"].to(device)
        descriptions = sample["description"].to(device)
        gt_bboxes = sample["bbox"].to(device)

        predicted_bboxes = model(images, descriptions)
        predicted_bboxes = cxcywh_to_xyxy(predicted_bboxes)
        loss = criterion_iou(gt_bboxes, predicted_bboxes)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Update the weights

        loss_array.append(loss.item())
    
    return loss_array

"""

def train_loop(model, student_model, data, optimizer, criterion_iou, device):
    model.train()
    loss_array = []


    for sample in tqdm(data, desc="Processing Training Dataset"):
    #for sample in data:
        images = sample["image"].to(device)
        descriptions = sample["description"].to(device)
        gt_bboxes = sample["bbox"].to(device)
        bbox_mask = sample["bbox_mask"].to(device)
        bbox_ratio = sample["bbox_ratio"].to(device)


        predicted_bboxes, all_attentions = model(images, descriptions)
        predicted_bboxes = cxcywh_to_xyxy(predicted_bboxes)

        #with torch.no_grad():
        #    _, all_attentions_mom = student_model(images, descriptions)

        rhos = {}
        for i, layer in enumerate(all_attentions):
            rhos[i] = spearmanr_batch(layer, bbox_mask)
        relative_rhos = compute_relative_rho(rhos)
        l_rac = rac_loss(relative_rhos, all_attentions, bbox_mask)
        #teacher_attentions = torch.stack(all_attentions)
        #student_attentions = torch.stack(all_attentions_mom)
        # be careful that the kl divergence is not symmetric, first student, second teacher
        #l_mrc = mrc_loss(student_attentions, teacher_attentions)

        #l_ar = l_rac + l_mrc

        #w_adw = 0.5 + 1 / (1 + math.exp(-l_ar))
        #w_odw = 0.5 + 1 / (1 + math.exp(bbox_ratio-1))
        w_odw = 1


        loss_second_part = criterion_iou(gt_bboxes, predicted_bboxes)
        #loss = l_ar + ((w_adw * w_odw) * loss_second_part)

        loss = l_rac/40 + loss_second_part

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Update the weights

        ### update the weights from teacher model to student one
        #update_momentum(model, student_model)

        loss_array.append(loss.item())
    
    return loss_array

"""
def eval_loop(model, dataloader, device):
    model.eval()
    all_ious = []
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating"):
        #for sample in dataloader:
            images = sample["image"].to(device)
            descriptions = sample["description"].to(device)
            gt_bboxes = sample["bbox"].to(device)

            predicted_bboxes = model(images, descriptions)
            predicted_bboxes = cxcywh_to_xyxy(predicted_bboxes)
            
            """
            for i, image in enumerate(images):
                image = image.to('cpu')
                image, pred_bbox, label_bbox, description = denormalize_data(image, predicted_bboxes[i].to('cpu').numpy(), gt_bboxes[i].to('cpu').numpy(), descriptions[i].to('cpu').numpy())

                path = f'/home/dec/uni/dl/visual-grounding/tests/image_{i}.png'
                compare_bbox(image, pred_bbox, label_bbox, save_path=path, caption=description, color1="green", color2="red")
            exit()
            """

            ious = calculate_IoU(gt_bboxes, predicted_bboxes)

            all_ious.extend(ious.tolist())

            correct += (ious > 0.5).sum().item()
            total += ious.size(0)

    accuracy = correct / total
    mean_iou = sum(all_ious) / len(all_ious)
    return mean_iou, accuracy
