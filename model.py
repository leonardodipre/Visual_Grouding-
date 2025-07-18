"""
Implemetation similar to TransVG architecture, using CLIP's visual and linguistic embeddings
instead of the papers's respective branches.
So we take the CLIP encodings and create a visual-linguistic fusion layer.

The paper's fusion layer is composed by two linear projection layers (one for each modality)
and a visual linguistic transformer (composed by a stack of 6 tranformer encoder layers).

The linear projection layers simply projects the different input dimensions of the two modality
to a embedding with the same channels dimension (cp=256).

To the visual encoder layer we will add a learnable embedding token ([REG]) which will then be used
to predict the actual bbox coordinates.
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import types
from transformer import PositionalEncoding, build_encoder_stack


class VisualLanguisticTranformer(nn.Module):

    def __init__(self, num_encoders, clip_model):
        super(VisualLanguisticTranformer, self).__init__()

        # modified_part = types.MethodType(new_part, where_to_attach_it_to)
        clip_model.visual.forward = types.MethodType(modified_visual_forward, clip_model.visual)
        clip_model.encode_text = types.MethodType(modified_encode_text, clip_model)
        self.clip_model = clip_model
        # we want to keep froze the CLIP's encoders.
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_model.eval()
        self.clip_model.float() # -> defaulf is float16, and pytorch default is float32. with .float() we make everything float32

        self.visual_projection_layer = nn.Linear(2048, 256)
        self.richer_visual_projection_layer = nn.Linear(3840, 256)
        self.textual_projection_layer = nn.Linear(1024, 256)
        self.segment_embed = nn.Embedding(3, 256)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        # max length: 49 visual + 77 desc + 77 root + 1 reg = 204
        self.positional_encoding = PositionalEncoding(d_model=256, seq_len=205, dropout=0.1)
        self.encoder_stack = build_encoder_stack(num_encoders=num_encoders, d_model=256, h=8, dropout=0.1, d_ff=2048)
        self.prediction_head = PredictionHead(hidden_dim=256)




    def forward(self, image, text, seg_ids, text_mask):
        #print(image.shape) # (batch_size, 3, 224, 224)
        image_embeds = self.clip_model.visual(image, concat=True) # (batch_size, 2048, 7, 7) # before modifying the visual (batch_size, 1024)
        text_embeds = self.clip_model.encode_text(text)  # (B, Lt+Lr, 1024)
        image_features_flattened = image_embeds.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1) # (batch_size, 49, 2048) or if we used concat=True (batch_size, 49, 3840)
        text_features_flattened = text_embeds # (batch_size, 77, 1024)
        if image_features_flattened.shape[2] == 2048:
            projected_visual = self.visual_projection_layer(image_features_flattened) # (batch_size, 49, 256)
        else:
            projected_visual = self.richer_visual_projection_layer(image_features_flattened) # (batch_size, 49, 256)
        projected_textual = self.textual_projection_layer(text_features_flattened)  # (B, Lt+Lr, 256)
        reg_token = torch.zeros(projected_textual.shape[0], 1, projected_textual.shape[-1], device=projected_textual.device)

        # dim -> the dimension over which the tensors are concatenated (can be differet among the tensors, the other shapes must match)
        seg_vis = torch.zeros(text.size(0), projected_visual.size(1), dtype=torch.long, device=text.device)
        seg_reg = torch.zeros(text.size(0), 1, dtype=torch.long, device=text.device)
        seg_all = torch.cat([seg_vis, seg_ids, seg_reg], dim=1)

        x = torch.cat((projected_visual, projected_textual, reg_token), dim=1) + self.segment_embed(seg_all)

        # add the positional encoding to the input
        x = self.positional_encoding(x)
        batch_size = projected_visual.shape[0]
        visual_mask = torch.ones(batch_size, projected_visual.shape[1], device=text.device)
        reg_mask = torch.ones(batch_size, 1, device=text.device)
        mask = torch.cat((visual_mask, text_mask, reg_mask), dim=1)
        mask = mask.unsqueeze(1).unsqueeze(2)

        output_encoder_stack, all_heads_attention = self.encoder_stack(x, mask)
        educated_reg_token = output_encoder_stack[:, -1, :] # now it's educated after having access to both visual and linguistic tokens # (batch_size, 256)
        predicted_bbox = self.prediction_head(educated_reg_token) # (batch_size, 4)
        return predicted_bbox, all_heads_attention



class PredictionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer   = nn.Linear(hidden_dim, 4) # 4 coordinates of the bbox

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x


def modified_visual_forward(self, x: torch.Tensor, concat=False):
    """
        Open AI official implementation, we removed the last attention pooling layer
        to keep more information.
        If fuse = True we concatenate from multiple ResNet layers to provide much richer information.
        - Multi-Scale visual information: first layers capture fine details while deeper layers capture higher-level features.
        The intermediate layers have shape:
            (batch_size, 256, 56, 56)
            (batch_size, 512, 28, 28)
            (batch_size, 1024, 14, 14)
            (batch_size, 2048, 7, 7) -> last layer before the attention pooling
        We want to concatenate them and end up with a tensor of (batch_size, 2048, 7, 7)
    """
    
    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x) # (batch_size, 256, 56, 56)
    x_layer1 = x
    x = self.layer2(x) # (batch_size, 512, 28, 28)
    x_layer2 = x
    x = self.layer3(x) # (batch_size, 1024, 14, 14)
    x_layer3 = x
    x = self.layer4(x) # (batch_size, 2048, 7, 7)

    if concat:
        pool = nn.AdaptiveAvgPool2d((7, 7)) # (batch_size, x, d, d) -> (batch_size, x, 7, 7)
        # (batch_size, 3840, 7, 7) [256+512+1024+2048]
        x = torch.cat((pool(x_layer1),
                      pool(x_layer2),
                      pool( x_layer3),
                      x), dim=1)
        

    # x = self.attnpool(x) <- removed attention pooling layer
    # now x can have shape (batch_size, 2048, 7, 7) or (batch_size, 3840, 7, 7)
    return x


def modified_encode_text(self, text):
    """
        We removed the last operation that was performing an argmax.
        Now instead of having a single embedding for a sentence,
        we have the embedding of each token of the sentence.
    """

    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    x = x @ self.text_projection

    return x