#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 23:13:23 2022

@author: menghanlin
"""

import torch
from torch.utils.data import Dataset
import pandas as pd



class pdb_dataset(Dataset):
    def __init__(self, dir_, transform = None):
        self.dir = dir_
        self.transform = transform
        self.protein_ligand = pd.read_pickle(self.dir + '/Ligand_binding/data/training_data.pkl')
    
    def __len__(self):
        return len(self.protein_ligand)

    def __getitem__(self, index):
        input_, output_ = self.protein_ligand[index]
        assert input_.shape == (23, 250, 250)
        assert output_.shape == (23, 250, 250)
        return input_, output_
    

if __name__ =='__main__':
    import os
    from torch.utils.data import DataLoader

    dir_ = os.getcwd()
    data = pdb_dataset(dir_)
    dataloader = DataLoader(
        data,
        batch_size=5,
        shuffle = True)
    for i, data in enumerate(dataloader):
        print(len(data[0]))
    # a = next(iter(dataloader))
    

