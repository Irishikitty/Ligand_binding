#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:16:55 2022

@author: menghanlin
"""

import pickle
import numpy as np
from preprocessing_utils import *
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min, pairwise_distances

# ========================================================================================


def distance_3D(x, y, axis=None):
    diff = np.array(x) - np.array(y)
    diff = diff ** 2
    return np.sqrt(np.sum(diff, axis=axis))


# ==============================================================================

def distance_to_atoms(candidate, verts, axis=None) :
    # return tuple(distance_3D(candidate, verts, axis=axis))
    return distance_3D(candidate, verts, axis=axis)

# ==============================================================================


def get_indices_of_k_smallest(arr, k = 206):

    idx = np.argpartition(arr.ravel(), k)
    counter = 0
    while (k+counter) < len(arr.ravel()):
        _, column_index = np.array(np.unravel_index(idx, arr.shape))[:, range(k+counter)]
        counter+=1
        if len(set(column_index.tolist())) == k:
            break
    # return np.array(np.unravel_index(idx, arr.shape))[:, range(k+counter)] #.transpose()
    return np.array(np.unravel_index(idx, arr.shape))[:, range(k+counter-1)][1].tolist()

# ==============================================================================

if __name__=="__main__":

    dataset = pickle.load(open("dataset_metal.pickle","rb"))
    # print(dataset)
    length_list = {'train': [], 'validation': [], 'test': []}  # {'train' : [[pdbid, chainID, ligand_atoms, axes, pos, ligand], [pdbid, chainID, ligand_atoms, axes, pos, ligand]]}
    for key, value in dataset.items():
        for i, data in enumerate(value):
            pdbid, chainID, ligand_atoms, axes, pos, ligand  = data
            file_name = pdbid.lower() + '.pdb'
            result = check_models(file_name)
            try:
                if result:
                    model_index = get_model_index(file_name, ligand, chainID, pos, result)
                    pdb_atoms4chainIDs = read_pdb(file_name, model_index, result)
                else:
                    pdb_atoms4chainIDs = read_pdb(file_name, result)
    
                min_dist_chain = {}
                for chain, atoms_data in pdb_atoms4chainIDs.items():
                    atomTypes, atomCoords = atoms_data
                    # _, dist = pairwise_distances_argmin_min(axes, np.array(atomCoords).T[1:].T)
                    dist = np.min(np.linalg.norm(axes - np.array(atomCoords).T[1:].T,2,axis=1))
                    min_dist_chain[chain] = dist
                # closest_chain = min(min_dist_chain, key=min_dist_chain.get)
                min_dist_chain_non_zero = {key: value for key, value in min_dist_chain.items() if value > 0}
                closest_chain = min(min_dist_chain_non_zero, key=lambda dict_key: min_dist_chain_non_zero[dict_key])
    
                if min_dist_chain[closest_chain] == 0:
                    raise Exception('check min function !!!')
                if closest_chain != chainID:
                    print(pdbid, chainID, pos, ligand)
                else:
                    assert closest_chain == chainID
                atomTypes_final, atomCoords_final = pdb_atoms4chainIDs[closest_chain]
                try:
                    assert len(atomCoords_final) >= 206
                    distance = pairwise_distances(axes, np.array(atomCoords_final).T[1:].T)
                    index_protein_atoms = list(set(get_indices_of_k_smallest(distance, 206)))
                    atomCoords_final_env = [atomCoords_final[index] for index in index_protein_atoms]
                    data.append(atomCoords_final_env)      # {'train' : [[pdbid, chainID, ligand_atoms, axes, pos, ligand, atomtypes, atomCoords], [pdbid, chainID, ligand_atoms, axes, pos, ligand, atomtypes, atomCoords]]}
                except AssertionError:
                    print('AssertionError raised: pdb id: ', pdbid, 'chain id: ', chainID, 'pos: ', pos, 'ligand: ', ligand)
            except:
                print(file_name)
    f = open("all_dataset_ligand_env.pickle","wb")
    pickle.dump(dataset,f)
    f.close()



