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

    def __init__(self, num_encoders, clip_model, use_concat=False, use_pyramid=False):
        super(VisualLanguisticTranformer, self).__init__()

        # modified_part = types.MethodType(new_part, where_to_attach_it_to)
        clip_model.visual.fpn = FPN()
        clip_model.visual.forward = types.MethodType(modified_visual_forward, clip_model.visual)
        clip_model.encode_text = types.MethodType(modified_encode_text, clip_model)
        self.clip_model = clip_model
        self.use_concat = use_concat
        self.use_pyramid = use_pyramid
        # we want to keep froze the CLIP's encoders.
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_model.eval()
        self.clip_model.float() # -> defaulf is float16, and pytorch default is float32. with .float() we make everything float32

        self.visual_projection_layer = nn.Linear(2048, 256)
        self.richer_visual_projection_layer = nn.Linear(3840, 256)
        self.visual_pyramid_projection = nn.Linear(1024, 256)
        self.textual_projection_layer = nn.Linear(1024, 256)
        self.positional_encoding = PositionalEncoding(d_model=256, seq_len=127, dropout=0.1)
        self.encoder_stack = build_encoder_stack(num_encoders=num_encoders, d_model=256, h=8, dropout=0.1, d_ff=2048)
        self.prediction_head = PredictionHead(hidden_dim=256)




    def forward(self, image, text):
        #print(image.shape) # (batch_size, 3, 224, 224)
        features = self.clip_model.visual(image, concat=self.use_concat, pyramid=self.use_pyramid)
        textual_mask = torch.where(text != 0, 1, 0)
        text_embeds = self.clip_model.encode_text(text)

        if self.use_pyramid:
            proj_pyramid = [F.adaptive_avg_pool2d(f, 7) for f in features]
            visual = torch.cat(proj_pyramid, dim=1)
            image_features_flattened = visual.flatten(2).permute(0, 2, 1)
            projected_visual = self.visual_pyramid_projection(image_features_flattened)
        else:
            image_features_flattened = features.flatten(2).permute(0, 2, 1)
            if self.use_concat:
                projected_visual = self.richer_visual_projection_layer(image_features_flattened)
            else:
                projected_visual = self.visual_projection_layer(image_features_flattened)

        text_features_flattened = text_embeds  # (batch_size, 77, 1024)
        projected_textual = self.textual_projection_layer(text_features_flattened)  # (batch_size, 77, 256)
        reg_token = torch.zeros(projected_textual.shape[0], 1, projected_textual.shape[-1]).to(projected_textual.device) # (batch_size, 1, 256)

        # dim -> the dimension over which the tensors are concatenated (can be differet among the tensors, the other shapes must match)
        x = torch.cat((projected_visual, projected_textual, reg_token), dim=1) # (batch_size, 49+77+1, 256)

        # add the positional encoding to the input
        x = self.positional_encoding(x)
        # for the mask to pass to the encoder stack we just use the attention mask of the textual tokens
        # this is because we don't have any padding for the image [we resize it to 224, 224 without putting any padding]
        # and the reg token doesn't also have padding of course. We concatenate these masks to create the one for the whole sequence
        batch_size = projected_visual.shape[0]
        visual_mask = torch.ones(batch_size, projected_visual.shape[1]).to(textual_mask.device)
        reg_mask = torch.ones(batch_size, 1).to(textual_mask.device)

        mask = torch.cat((visual_mask, textual_mask, reg_mask), dim=1) # (batch_size, 127)

        mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, 127)

        output_encoder_stack, all_heads_attention = self.encoder_stack(x, mask) # (batch_size, 127, 256)
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


class FPN(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        p4 = self.lat[3](c4)
        p3 = self.lat[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.lat[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        p4, p3, p2, p1 = [s(f) for s, f in zip(self.smooth, [p4, p3, p2, p1])]
        return [p1, p2, p3, p4]


def modified_visual_forward(self, x: torch.Tensor, concat=False, pyramid=False):
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
    c1 = self.layer1(x)
    c2 = self.layer2(c1)
    c3 = self.layer3(c2)
    c4 = self.layer4(c3)

    if concat:
        pool = nn.AdaptiveAvgPool2d((7, 7))
        return torch.cat((pool(c1), pool(c2), pool(c3), c4), dim=1)

    if pyramid:
        return self.fpn([c1, c2, c3, c4])

    return c4


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
