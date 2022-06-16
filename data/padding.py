#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:01:47 2022

@author: menghanlin
"""
from sklearn.metrics import pairwise_distances
import numpy as np
import torch
# import seaborn as sns

def data_to_0_1(sample):
    max = torch.max(sample)
    min = torch.min(sample)

    out = 2*sample/(max-min) - 1
    return out

def data_toTensor(ligand, atoms, dim_max=250):
    '''
    3D coordinates of input (ligand + atom)
    Ouptut tensor with shape [23, dim_sum = 250, 250]

    Input:
        :param ligand:
        :param atoms:
        :param dim_max:
    Output:

    '''
    numAtoms, _= np.array(atoms).shape
    numLigandAtoms, _ = np.array(ligand).shape
    
    A_coordinates = pairwise_distances([[x,y,z] for [_,x,y,z] in ligand] + [[x,y,z] for [i,x,y,z] in atoms])
    types = [i for [i,_,_,_] in atoms]
    container = torch.zeros(22, *A_coordinates.shape)
    
    for i,c in enumerate(types):
        for j in range(len(types)):
            container[c-1,i,j] +=1
            container[c-1,j,i] +=1
    container[container > 0] = 1 # as indicator
    
    # build tensor
    A_coordinates = torch.from_numpy(A_coordinates.reshape(-1, *A_coordinates.shape))
    A_coordinates = torch.cat([A_coordinates, container], dim = 0)
    A_coordinates[22,:numLigandAtoms,:numLigandAtoms] = 1

    # move blocks
    out = torch.zeros(23, dim_max, dim_max) # first channel container
    out[:, :numLigandAtoms, :numLigandAtoms] += A_coordinates[:,:numLigandAtoms, :numLigandAtoms] # upper left
    out[:, 50:50+numAtoms, 50:50+numAtoms] += A_coordinates[:,numLigandAtoms:, numLigandAtoms:]   # lower right
    out[:, :numLigandAtoms, 50:50+numAtoms] += A_coordinates[:,:numLigandAtoms,numLigandAtoms:]   # rectangle upper
    out[:, 50:50+numAtoms, :numLigandAtoms] += A_coordinates[:,numLigandAtoms:,:numLigandAtoms] # rectangle lower

    return out

# input_ = data_toTensor(starting_ligand, atoms)
# output = data_toTensor(true_ligand, atoms)

# sns.heatmap(input_[1])
# sns.heatmap(output[0])


