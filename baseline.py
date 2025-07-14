import torch
import clip
import torchvision.transforms.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


def yolo_inference(model, image):
    # we will pass the model to not load it each time
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # apparentely yolov5 doesnt want the image to be a tensor... "don't pass a tensor. Pass anything else from the example. Tensors are for training only."
    # tensor batch of images to numpy arrays using tensor.numpy(), and then pass these numpy arrays to the YOLOv5 model for inference
    
    # Inference
    image = F.to_pil_image(image)
    results = model(image)
 
    # Results
    
    """
    results: xmin    ymin    xmax   ymax  confidence  class    name
    """
    bboxes = results.xyxy[0].cpu().numpy()

    bboxes = [bbox[:4] for bbox in bboxes] # take only the coordinates of the bbox


    
    cropped_images = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        # get the content of each bbox
        cropped_img = image.crop((xmin, ymin, xmax, ymax))
        cropped_images.append(cropped_img)
    
    #crops = results.crop(save=False)  # cropped detections dictionary -> directly from yolo. if save=True the cropped elements will be saved in runs/detect/exp/crops

    # we return the content of the bboxes and the coordinates of them
    return cropped_images, bboxes


def clip_inference(model, preprocess, regions, bboxes, description):
    # to not load it everytime we pass it
    #model, preprocess = clip.load("RN50", device=device)

    if regions == []: # in case yolo did not predict anything
        dummy_bbox = [0, 0, 0, 0]
        return dummy_bbox
    # Preprocess the images
    images = [preprocess(region).unsqueeze(0).to(device) for region in regions]
    image_batch = torch.cat(images, dim=0)
    ## already done
    #description = "a photo of " + description
    #description = clip.tokenize(description).to(device) # tokenize the description and move on gpu

    with torch.no_grad():
        description = description.unsqueeze(0).to(device) # we squeezed in the Dataset
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(description)

        similarity = image_features @ text_features.T
        probs = similarity.softmax(dim=0).cpu()
        max_prob_index = torch.argmax(probs).numpy()
    
    pred_bbox = bboxes[max_prob_index]

    return pred_bbox
