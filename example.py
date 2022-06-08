#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 08:48:09 2022

@author: menghanlin
"""
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import pdb_dataset
import torch.optim as optim
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from axial_attention import AxialAttention, AxialPositionalEmbedding

BATCH_SIZE = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = pdb_dataset()
dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle = True)

attn = AxialAttention(
    dim = 23,               # embedding dimension
    dim_index = 1,         # where is the embedding dimension
    dim_heads = 23,        # dimension of each head. defaults to dim // heads if not supplied
    heads = 1,             # number of heads for multi-head attention
    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
    sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
)


model = nn.Sequential(attn, attn, attn, attn, torch.nn.Conv2d(23, 1, 1))
model.load_state_dict(torch.load('model.ckpt',map_location=torch.device('cpu'))['model_state_dict'])

epoch, model_state_dict, optimizer_state_dict, loss = torch.load('model.ckpt',map_location=torch.device('cpu')).values()
sample_input, sample_output, sample_input_lens= next(iter(dataloader))
pred = model(sample_input)

index = 1
sns.heatmap(pred.detach()[index,0])
sns.heatmap(sample_output[index,0])

pred.detach()[index,0,199:200]
sample_output[index,0,199:200]
plt.hist(pred.detach()[index,0,199:200])
plt.hist(sample_output[index,0,199:200])




