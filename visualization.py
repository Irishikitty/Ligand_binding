#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:28:19 2022

@author: menghanlin
"""

# Heatmap after rotation
import dataloader
import os
from torch.utils.data import DataLoader
import seaborn as sns

dir_ = os.getcwd()
data = pdb_dataset(dir_)
dataloader = DataLoader(
    data,
    batch_size=64,
    shuffle = True)
a = next(iter(dataloader))
    
sns.heatmap(a[0][1,0],vmax = 50)
sns.heatmap(a[1][1,0],vmax = 50)
