#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 23:13:23 2022

@author: menghanlin
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.spatial.transform import Rotation as R
from data_preprocessing import data_preprocessing_tensor
from sklearn.metrics import pairwise_distances
import numpy as np



class pdb_dataset(Dataset):
    def __init__(self, dir_, env_size = 200, ligand_size = 50, transform = None):
        self.dir = dir_
        self.input_size = env_size + ligand_size
        self.transform = transform
        self.protein_ligand = pd.read_pickle(self.dir)
    
    def __len__(self):
        return len(self.protein_ligand)

    def __getitem__(self, index):
        # Generate true tensor
        train_sample = self.protein_ligand[index]
        types = [i[0] for i in train_sample[3]] + [i[0] for i in train_sample[2]]
        axes = [i[1:] for i in train_sample[3]] + [i[1:] for i in train_sample[2]]
        pwdist = pairwise_distances(axes)
        input_len = len(axes)
        
        output_ = torch.zeros(1, self.input_size, self.input_size)
        output_[0, :input_len, :input_len] += pwdist
        
        
        # Generate random ligand position
        r = R.random().as_euler('zxy', degrees=True)
        rotation_mx = R.from_euler('zxy', r).as_matrix()
        shift = np.random.rand(1) * 18 #???
        rot_ligand = [rotation_mx @ np.array(i[1:]).reshape(3,1)+shift for i in train_sample[2]]
        rot_axes = [i[1:] for i in train_sample[3]] + [i.reshape(-1,3).tolist()[0] for i in rot_ligand]
        pwdist = pairwise_distances(rot_axes)
        
        input_ = torch.zeros(1, self.input_size, self.input_size)
        input_[0, :input_len, :input_len] += pwdist
        
        # Add type channels
        tensor_stack = data_preprocessing_tensor(types)
        
        input_ = torch.cat([input_, tensor_stack], dim = 0)
        output_ = torch.cat([output_, tensor_stack], dim = 0)
        
        assert input_.shape == (23, 250, 250)
        assert output_.shape == (23, 250, 250)
        
        return input_, output_
    
    
    

if __name__ =='__main__':
    import os
    from torch.utils.data import DataLoader

    dir_ = os.getcwd()
    data = pdb_dataset(dir_+'/ligand_env_coords_subset_zn.pkl')
    dataloader = DataLoader(
        data,
        batch_size=10,
        shuffle = True)
    for i, data in enumerate(dataloader):
        print(len(data[0]))
    # a = next(iter(dataloader))
    

