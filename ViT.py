#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:40:54 2022

@author: menghanlin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# step 1: convert image to embedding sequences
def image2emb_naive(image, patch_size, weight):
    # image size: [bs, channel, H, W]
    patch = F.unfold(image, kernel_size = patch_size, stride=patch_size).transpose(-1, -2)
    patch_embedding = patch @ weight
    return patch_embedding


def image2emb_conv(image, kernel, stride):
    conv_output = F.conv2d(image, kernel, stride = stride) #[bs, oc, oh, ow]
    bs, oc, oh, ow = conv_output.shape
    patch_embedding = conv_output.reshape((bs, oc, oh*ow)).transpose(-1,-2) # oh*ow是序列长度
    return patch_embedding

bs, ic, image_h, image_w = 1, 3, 8, 8
# bs, ic, image_h, image_w = 1, 3, 250, 250
patch_size = 4
model_dim = 8
max_num_token = 16
num_classes = 10
label = torch.randint(10, (bs,))

patch_depth = patch_size * patch_size * ic # 每个patch大小
image = torch.randn(bs, ic, image_h, image_w)
weight = torch.randn(patch_depth, model_dim) # model_dim是输出通道数目，patch_depth是卷积核的面积*输入通道数

patch_embedding_naive = image2emb_naive(image, patch_size, weight)
kernel = weight.transpose(0,1).reshape((-1, ic, patch_size, patch_size)) # oc * ic * kh * kw
patch_embedding_conv = image2emb_conv(image, kernel, patch_size)

# print(patch_embedding_naive.shape)
# print(patch_embedding_conv.shape) # [bs = 1, seq = 4, model_dim = 8]

# step 2: add classification token
cls_token_embedding = torch.randn(bs, 1, model_dim, requires_grad=True) # 1 is length
token_embedding = torch.cat([cls_token_embedding, patch_embedding_conv], dim = 1) # 在位置上cat

# step 3: add position embedding
position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
seq_len = token_embedding.shape[1]
position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1]) #position embedding复制bs份
token_embedding += position_embedding

#step 4: pass embedding to transormer encoder 
encoder_layer = nn.TransformerEncoderLayer(d_model = model_dim, nhead = 8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 6)
encoder_output = transformer_encoder(token_embedding)

# step 5: do classification
cls_token_output = encoder_output[:, 0, :]
linear_layer = nn.Linear(model_dim, num_classes)
logits = linear_layer(cls_token_output)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, label)
print(loss)





