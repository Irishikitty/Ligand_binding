#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:00:39 2022

@author: menghanlin
"""
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import pdb_dataset
import torch.optim as optim
import os
import seaborn as sns
from axial_attention import AxialAttention, AxialPositionalEmbedding

BATCH_SIZE = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = pdb_dataset()
dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle = True)
# encoder_layer = nn.TransformerEncoderLayer(250, 5)
# encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)


img = torch.randn(1, 23, 250, 250)

attn = AxialAttention(
    dim = 23,               # embedding dimension
    dim_index = 1,         # where is the embedding dimension
    dim_heads = 23,        # dimension of each head. defaults to dim // heads if not supplied
    heads = 1,             # number of heads for multi-head attention
    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
    sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
)

pos_emb = AxialPositionalEmbedding(
    dim = 23,
    shape = (250, 250)
)

model = nn.Sequential(attn, attn, attn).to(device)

# pos_emb(img)
# attn(img) 

optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.L1Loss()
loss_lst = []

for epoch in range(2):
    for i, j in enumerate(dataloader):
        inputs, outputs = j
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        
        optimizer.zero_grad()
        
        pred = model(inputs)
        loss = criterion(outputs,pred)
        loss.backward()
        optimizer.step()
        
        
        with torch.no_grad():
            # sns.heatmap(pred[0,0])
            print(loss.detach())
            loss_lst.append(loss.detach())
        
        
torch.save({'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimize.state_dict(),
    'loss': loss.detach(),
    }, 'model.ckpt')
