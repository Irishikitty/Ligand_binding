#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:01:47 2022

@author: menghanlin
"""
from sklearn.metrics import pairwise_distances
import seaborn as sns

# numAtoms, _= np.array(atoms).shape
# numLigandAtoms, _ = np.array(true_ligand).shape
# dims = numAtoms + numLigandAtoms
# assert np.array(true_ligand).shape == starting_ligand.shape

# A_coordinates = pairwise_distances([[x,y,z] for [i,x,y,z] in starting_ligand] + [[x,y,z] for [i,x,y,z] in atoms])
# types = [i for [i,x,y,z] in atoms]
# container = torch.zeros(22, *A_coordinates.shape)

# for i,c in enumerate(types):
#     for j in range(len(types)):
#         container[c-1,i,j] +=1
#         container[c-1,j,i] +=1
# container[container > 0] = 1 # as indicator
# A_coordinates = torch.from_numpy(A_coordinates.reshape(-1, *A_coordinates.shape))
# A_coordinates = torch.cat([A_coordinates, container], dim = 0)
# A_coordinates[22,:numLigandAtoms,:numLigandAtoms] = 1


# # move blocks
# out = torch.zeros(23, dim_max, dim_max) # first channel container
# out[:, :numLigandAtoms, :numLigandAtoms] += A_coordinates[:,:numLigandAtoms, :numLigandAtoms] # upper left
# out[:, 50:50+numAtoms, 50:50+numAtoms] += A_coordinates[:,numLigandAtoms:, numLigandAtoms:]   # lower right
# out[:, :numLigandAtoms, 50:50+numAtoms] += A_coordinates[:,:numLigandAtoms,numLigandAtoms:]   # rectangle upper
# out[:, 50:50+numAtoms, :numLigandAtoms] += A_coordinates[:,numLigandAtoms:,:numLigandAtoms]   # rectangle lower

# # build fake A and True B
# B_coordinates = pairwise_distances([[x,y,z] for [i,x,y,z] in true_ligand] + [[x,y,z] for [i,x,y,z] in atoms])

# input_ = torch.zeros(1, dim_max, dim_max) # first channel container
# input_[0, :numLigandAtoms, :numLigandAtoms] += A_coordinates[:numLigandAtoms, :numLigandAtoms] # upper left
# input_[0, 50:50+numAtoms, 50:50+numAtoms] += A_coordinates[numLigandAtoms:, numLigandAtoms:]   # lower right
# input_[0, :numLigandAtoms, 50:50+numAtoms] += A_coordinates[:numLigandAtoms,numLigandAtoms:]   # rectangle upper
# input_[0, 50:50+numAtoms, :numLigandAtoms] += A_coordinates[:numLigandAtoms,numLigandAtoms:].T # rectangle lower

# # stack other channels
# sns.heatmap(channel_1[0])


# output = torch.zeros(1, dim_max, dim_max) # first channel container
# output[0, :numLigandAtoms, :numLigandAtoms] += B_coordinates[:numLigandAtoms, :numLigandAtoms]
# output[0, 50:50+numAtoms, 50:50+numAtoms] += B_coordinates[numLigandAtoms:, numLigandAtoms:]
# output[0, :numLigandAtoms, 50:50+numAtoms] += B_coordinates[:numLigandAtoms,numLigandAtoms:]
# output[0, 50:50+numAtoms, :numLigandAtoms] += B_coordinates[:numLigandAtoms,numLigandAtoms:].T


def data_toTensor(ligand, atoms):
    
    numAtoms, _= np.array(atoms).shape
    numLigandAtoms, _ = np.array(ligand).shape
    
    A_coordinates = pairwise_distances([[x,y,z] for [i,x,y,z] in ligand] + [[x,y,z] for [i,x,y,z] in atoms])
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

input_ = data_toTensor(starting_ligand, atoms)
output = data_toTensor(true_ligand, atoms)

sns.heatmap(input_[1])
sns.heatmap(output[0])


