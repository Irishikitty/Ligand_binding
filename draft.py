#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:02:31 2022

@author: menghanlin
"""

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sn
train= pd.read_pickle('ligand_env_coords.pkl')
temp = train[100]

env_size = 200
ligand_size = 50
batch_size = 1
type_size = 22

types = [i[0] for i in temp[3]] + [i[0] for i in temp[2]]
axes = [i[1:] for i in temp[3]] + [i[1:] for i in temp[2]]

axes = torch.FloatTensor(axes)
input_len = len(axes)
pwdist = pairwise_distances(axes)

input_size = env_size + ligand_size
src_input = torch.zeros(batch_size, input_size, input_size)

# Data preprocessing ==========================
src_input[0,:input_len, :input_len] += pwdist
type_input = types

sn.heatmap(pwdist)

import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import numpy as np
np.fill_diagonal(pwdist,0)
linkage = hc.linkage(sp.distance.squareform(pwdist), method='average')
sn.clustermap(pwdist, row_linkage=linkage, col_linkage=linkage)

strip = pwdist[:200, 200:]
sn.heatmap(strip)
sn.clustermap(strip)

fig = plt.figure()
ax = plt.axes(projection = '3d')
z = [z for x,y,z in axes]
x = [x for x,y,z in axes]
y = [y for x,y,z in axes]

ax.scatter(x[:200],y[:200],z[:200],'green')
ax.scatter(x[200:],y[200:],z[200:], c='r')
ax.set_title('3d')
plt.show()

# Encoder ============================

    # First pwdist mx (batch_size, 250, 250, emb)
    # row-wise self attention 
    # TODO: + type encoding 
    # TODO: + positional encoding?
    

    # column-wise self attention
    # output: (batch_size, n_head, )
    
encoder_layer = nn.TransformerEncoderLayer(250, 5)
# encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

optimizer = optim.Adam(encoder_layer.parameters(), lr = 0.0001)
loss_lst = []


for i in range(90000):
    
    temp = train[i]
    types = [i[0] for i in temp[3]] + [i[0] for i in temp[2]]
    axes = [i[1:] for i in temp[3]] + [i[1:] for i in temp[2]]
    
    if len(axes) <= 250:
        axes = torch.FloatTensor(axes)
        input_len = len(axes)
        pwdist = pairwise_distances(axes)
    
        src_input = torch.zeros(input_len, input_size)
    
        # Data preprocessing ==========================
        src_input[:, :input_len] += pwdist
        type_input = types
        
        mask_ = torch.zeros(input_len,250)
        mask_[:,input_len:] += 1
        
        optimizer.zero_grad()
        outputs = encoder_layer(src_input, src_key_padding_mask=mask_) #[batch_size, sequence_length]
        loss = torch.sum(torch.abs(outputs[:,:input_len,200:input_len] - src_input[:,:input_len,200:input_len])) 
        loss.backward()
        optimizer.step()
        
        loss_lst.append(loss.detach())
        print(loss.detach())
    
    
plt.plot(loss_lst)   
    
    
# use ligand len < 50 as subsample
train_sample = pd.read_pickle('ligand_env_coords.pkl')
idx = [len(i[2])<=50 for i in train_sample]
subsample = [train_sample[i] for i,j in enumerate(idx) if j==True]

import pickle
with open('ligand_env_coords_subset.pkl', 'wb') as handle:
    pickle.dump(subsample, handle)   
    
    
    
    
    
    
    
    
    
    