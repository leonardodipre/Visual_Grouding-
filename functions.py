import math
import torch
import torch.nn.functional as F
import copy
from baseline import yolo_inference, clip_inference
from util import calculate_IoU, cxcywh_to_xyxy, compare_bbox, draw_bbox, denormalize_data, rac_loss, mrc_loss, spearmanr_batch, compute_relative_rho, update_momentum
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

W_ODW_BETA = 0.5

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


def train_loop(model, student_model, data, optimizer, criterion_iou, device, selected_loss):
    model.train()
    loss_array = []
    # if we selected the attention regulated loss we do some additional things
    # we set it to True for comodity

    total_losses = []
    rac_losses = []
    mrc_losses = []
    ar_losses = []
    adw_weights = []
    odw_weights = []
    bbox_losses = []


    if selected_loss == 'att_reg':
        selected_loss = True
    else:
        selected_loss = False

    for sample in tqdm(data, desc="Processing Training Dataset"):
    #for sample in data:
        images = sample["image"].to(device)
        descriptions = sample["description"].to(device)
        gt_bboxes = sample["bbox"].to(device)
        bbox_mask = sample["bbox_mask"].to(device)
        if selected_loss:
            bbox_ratio = sample["bbox_ratio"].to(device)


        predicted_bboxes, all_attentions = model(images, descriptions)
        predicted_bboxes = cxcywh_to_xyxy(predicted_bboxes)
        if selected_loss:
            with torch.no_grad():
                _, all_attentions_mom = student_model(images, descriptions)

            rhos = {}
            for i, layer in enumerate(all_attentions):
                rhos[i] = spearmanr_batch(layer, bbox_mask)
            relative_rhos = compute_relative_rho(rhos)
            l_rac = rac_loss(relative_rhos, all_attentions, bbox_mask)
            teacher_attentions = torch.stack(all_attentions)
            student_attentions = torch.stack(all_attentions_mom)
            # be careful that the kl divergence is not symmetric, first student, second teacher
            l_mrc = mrc_loss(student_attentions, teacher_attentions)

            l_ar = (l_rac/49) + l_mrc

            w_adw = 0.5 + 1 / (1 + math.exp(-l_ar))
           
            #w_odw = 0.5 + 1 / (1 + math.exp(bbox_ratio.mean()-1)) # now we take the mean... Will be interesting to multiply the single example loss for the single bbox ratio
            w_odw = 0.5 + 1 / (1 + math.exp(W_ODW_BETA * (bbox_ratio.mean() - 1)))

            loss_second_part = criterion_iou(gt_bboxes, predicted_bboxes)
            loss = l_ar + ((w_adw * w_odw) * loss_second_part)

            rac_losses.append(l_rac.item())
            mrc_losses.append(l_mrc.item())
            ar_losses.append(l_ar.item())
            adw_weights.append(w_adw)
            odw_weights.append(w_odw)
            bbox_losses.append(loss_second_part.item())
        else:
            loss = criterion_iou(gt_bboxes, predicted_bboxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Update the weights

        if selected_loss:
            ### update the weights from teacher model to student one
            update_momentum(model, student_model)
            total_losses.append(loss.item())

    metrics = {"loss": np.mean(total_losses)}

    if selected_loss:
        metrics.update(
            {
                "rac_loss": np.mean(rac_losses),
                "mrc_loss": np.mean(mrc_losses),
                "att_reg_loss": np.mean(ar_losses),
                "w_adw": np.mean(adw_weights),
                "w_odw": np.mean(odw_weights),
                "bbox_loss": np.mean(bbox_losses),
            }
        )

    return metrics


def train_loop_attbalance(model, data, optimizer, criterion_iou, lambda_att=1.0, device="cuda"):
    """Training loop with root-guided attention balance."""
    model.train()
    losses = []
    for sample in tqdm(data, desc="Processing Training Dataset"):
        images = sample["image"].to(device)
        text = sample["text"].to(device)
        seg_ids = sample["seg_ids"].to(device)
        text_mask = sample["text_mask"].to(device)
        root_mask = sample["root_mask"].to(device)
        gt_bboxes = sample["bbox"].to(device)
        bbox_mask = sample["bbox_mask"].to(device)

        pred_box, attn = model(images, text, seg_ids, text_mask)
        pred_box = cxcywh_to_xyxy(pred_box)
        loss_bbox = criterion_iou(gt_bboxes, pred_box)

        A = attn[-1].mean(1)
        S = A.size(-1)
        Lv = S - text.size(1) - 1
        reg_idx = S - 1
        reg2vis = A[:, reg_idx, :Lv]
        root2vis = A[:, root_mask.bool(), :Lv].mean(1)

        gated = reg2vis * (1 + model.gamma * root2vis)
        gated = gated / gated.sum(dim=1, keepdim=True)
        loss_att = F.binary_cross_entropy(gated, bbox_mask.float(), weight=bbox_mask.float())

        loss = loss_bbox + lambda_att * loss_att

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return {"loss": np.mean(losses)}

        
    
    



def eval_loop(model, dataloader, device):
    model.eval()
    all_ious = []
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating"):
        #for sample in dataloader:
            images = sample["image"].to(device)
            if "text" in sample:
                text = sample["text"].to(device)
                seg_ids = sample["seg_ids"].to(device)
                text_mask = sample["text_mask"].to(device)
                predicted_bboxes, _ = model(images, text, seg_ids, text_mask)
            else:
                descriptions = sample["description"].to(device)
                predicted_bboxes, _ = model(images, descriptions)
            gt_bboxes = sample["bbox"].to(device)

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
