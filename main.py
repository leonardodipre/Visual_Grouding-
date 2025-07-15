
import argparse
from util import *
from functions import train_loop, eval_loop, eval_loop_baseline
import warnings
from tqdm import tqdm
from model import VisualLanguisticTranformer
import clip
import torch
import copy
import numpy as np
import torchvision
import pandas as pd
import ast

warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = 'cuda'

def main(args):

    start_checkpoint = args.start_checkpoint
    end_checkpoint = args.end_checkpoint
    batch_size = args.batch_size
    n_epochs = args.epochs
    learning_rate = args.learning_rate
    optimizers = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW, 'sgd': torch.optim.SGD}

    selected_loss = args.criterion

    verbose = args.verbose

    
    annotations_file_path = '/home/rtx/deep_learning/dataset/refcocog/annotations/instances.json'
    pickle_file_path = '/home/rtx/deep_learning/dataset/refcocog/annotations/refs(umd).p'
    

    whole_df = create_merged_df(pickle_file_path, annotations_file_path)
    
    #whole_df = pd.read_csv('dataset/refcocog_augmented.csv')
    #whole_df['bbox'] = whole_df['bbox'].apply(ast.literal_eval)

    # split the whole dataframe in train, val, test
    train_df = whole_df.loc[whole_df['split'] == 'train']
    val_df   = whole_df.loc[whole_df['split'] == 'val']
    test_df  = whole_df.loc[whole_df['split'] == 'test']

    keep_aspect_ratio = True # -> if False stretches the image to 224x224
    image_transform = modified_clip_preprocess(keep_aspect_ratio)
    bbox_transform = lambda bbox, orig, new: resize_bbox(bbox, orig, new, keep_aspect_ratio)
    tokenizer = clip.tokenize
    prefix_description =  "find the region that corresponds to the description "
    train_dataset = VisualGroundingRefcocog(train_df, tokenizer, prefix_description, image_transform, bbox_transform)
    val_dataset = VisualGroundingRefcocog(val_df, tokenizer, prefix_description, image_transform, bbox_transform)
    test_dataset = VisualGroundingRefcocog(test_df, tokenizer, prefix_description, image_transform, bbox_transform)# has 5024 elements
    
    train_dataloader = get_dataloader(train_dataset, batch_size)
    val_dataloader = get_dataloader(val_dataset, batch_size)
    test_dataloader = get_dataloader(test_dataset, batch_size)

    clip_model, _ = clip.load("RN50", device=DEVICE)
    num_encoders = 6
    model = VisualLanguisticTranformer(num_encoders, clip_model)
    # we apply the init_weights function to initialize the projection layers -> speed up training
    # we start with better weights.
    model.apply(init_weights)
    model.to(DEVICE)
    optimizer = optimizers[args.optimizer](model.parameters(), lr=learning_rate)
    if verbose:
        print(f'start_checkpoint {start_checkpoint}, and saving on {end_checkpoint}')
    if start_checkpoint != "none":
        model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, f"bin/checkpoint_{start_checkpoint}.pth")
        if verbose:
            print(f'{start_checkpoint} correctly loaded!')
    else:
        start_epoch = 0
    #mean_iou, accuracy = eval_loop(model, test_dataloader, device=DEVICE)
    #print(f'mean iou on test set is {mean_iou} --- accuracy = {accuracy}')
    #exit()
    total_epochs = start_epoch + n_epochs

    ### For the MRC loss
    if selected_loss == 'att_reg':
        momentum_model = copy.deepcopy(model)
        for param in momentum_model.parameters():
            param.requires_grad = False  # no backprop
    else:
        momentum_model = None

    for epoch in tqdm(range(start_epoch +1 , total_epochs+1)):
    #for epoch in range(start_epoch +1 , total_epochs+1):
        loss = train_loop(model, momentum_model, train_dataloader, optimizer, criterion_iou, device=DEVICE, selected_loss=selected_loss)
        if verbose:
            print(f'loss at epoch {epoch} is {np.asarray(loss).mean()}')
        if epoch % 2 == 0: # We check the performance every 3 epochs
            mean_iou, accuracy = eval_loop(model, val_dataloader, device=DEVICE)
            if verbose:
                print(f'mean_iou at epoch {epoch} = {mean_iou} --- accuracy = {accuracy}')
    if end_checkpoint != "none":
        save_checkpoint(model, optimizer, total_epochs, loss, f"bin/checkpoint_{end_checkpoint}.pth")
    mean_iou, accuracy = eval_loop(model, test_dataloader, device=DEVICE)
    print(f'mean iou on test set is {mean_iou} --- accuracy = {accuracy}')



def evaluate_baseline(data, device, modified_preprocess=None):
    clip_model, clip_preprocess = clip.load("RN50", device=device)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
    if modified_preprocess is None:
        modified_preprocess = clip_preprocess
    
    accuracy, mean_iou = eval_loop_baseline(clip_model, modified_preprocess, yolo_model, data)

    return accuracy, mean_iou


if __name__ == "__main__":
     # Create the parser
    parser = argparse.ArgumentParser(description='Visual Grounding using CLIP and Transformer one stage approach.')
    
    # Add arguments
    parser.add_argument('--batch_size', default="32", type=int, help='batch size of training')
    parser.add_argument('--epochs', default="50", type=int, help='number of epochs')
    parser.add_argument('--optimizer', default="AdamW", help="select 'Adam' or 'sgd' or 'AdamW'")
    parser.add_argument('--learning_rate', default="0.0001", type=float, help='learning rate of the model')
    parser.add_argument('--patience', default="5", type=int, help='patience of the model')
    parser.add_argument('--criterion', default="normal", help="select 'normal' or 'att_reg'")
    parser.add_argument('--start_checkpoint', default="none", help="name of the checkpoint to be loaded")
    parser.add_argument('--end_checkpoint', default="none", help="name of the checkpoint to be saved")
    parser.add_argument('--verbose', default=False, type=str2bool, help="verbose or not")


    
    # Parse the arguments
    args = parser.parse_args()
    main(args)

    