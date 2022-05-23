#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:45:51 2022

@author: menghanlin

Data preprocessing and subset selection
"""
from collections import Counter
import pickle
import torch


# # use ligand len < 50 as subsample
# train_sample = pd.read_pickle('ligand_env_coords.pkl')
# idx = [len(i[2])<=50 for i in train_sample]
# subsample = [train_sample[i] for i,j in enumerate(idx) if j==True]


# # count the number of each single atom
# atoms = [i[1] for i in subsample if len(i[2])==1]
# Counter(atoms) # 11,958
# subsample_zn = [i for i in subsample if i[1] == 'ZN']


# with open('ligand_env_coords_subset_zn.pkl', 'wb') as handle:
#     pickle.dump(subsample_zn, handle)   

def data_preprocessing_tensor(types):
    container = torch.zeros(22, 250, 250)
    for i,c in enumerate(types):
        for j in range(250):
            container[c-1,i,j] +=1
            container[c-1,j,i] +=1
    return container

