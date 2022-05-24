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
import os
import pandas as pd
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import pairwise_distances
import numpy as np

np.random.seed(10)
input_size = 250

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

dir_ = os.getcwd()
protein_ligand = pd.read_pickle(dir_+ '/Ligand_binding/ligand_env_coords_subset_zn.pkl')
training_data=[]
for index in range(len(protein_ligand)):
                   
    train_sample = protein_ligand[index]
    types = [i[0] for i in train_sample[3]] + [i[0] for i in train_sample[2]]
    axes = [i[1:] for i in train_sample[3]] + [i[1:] for i in train_sample[2]]
    pwdist = pairwise_distances(axes)
    input_len = len(axes)
    
    output_ = torch.zeros(1, input_size, input_size)
    output_[0, :input_len, :input_len] += pwdist
    
    
    # Generate random ligand position
    r = R.random().as_euler('zxy', degrees=True)
    rotation_mx = R.from_euler('zxy', r).as_matrix()
    shift = np.random.rand(1) * 18 #???
    rot_ligand = [rotation_mx @ np.array(i[1:]).reshape(3,1)+shift for i in train_sample[2]]
    rot_axes = [i[1:] for i in train_sample[3]] + [i.reshape(-1,3).tolist()[0] for i in rot_ligand]
    pwdist = pairwise_distances(rot_axes)
    
    input_ = torch.zeros(1, input_size,input_size)
    input_[0, :input_len, :input_len] += pwdist
    
    # Add type channels
    tensor_stack = data_preprocessing_tensor(types)
    
    input_ = torch.cat([input_, tensor_stack], dim = 0)
    output_ = torch.cat([output_, tensor_stack], dim = 0)
    
    assert input_.shape == (23, 250, 250)
    assert output_.shape == (23, 250, 250)
    print(index)
    training_data.append([input_, output_])
    
with open('training_data.pkl', 'wb') as handle:
    pickle.dump(training_data, handle)  