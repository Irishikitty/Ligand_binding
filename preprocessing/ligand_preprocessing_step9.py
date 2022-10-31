import pickle
import numpy as np
from tqdm import tqdm

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

    dataset = pickle.load(open("dataset.pickle","rb"))
    # print(dataset)
    length_list = {'train': [], 'validation': [], 'test': []}  # {'train' : [[pdbid, chainID, ligand_atoms, axes, pos, ligand], [pdbid, chainID, ligand_atoms, axes, pos, ligand]]}
    closest_chain_inconsistent_list = []
    assertError_file_list = []

    dataset_env_atoms = {}

    pdb_atoms4chainIDs_empty_data_list = []
    for key, value in dataset.items():
        # if key == 'train':
        #     values = value[94098:]
        data_deque = deque()
        for data in tqdm(value):
            pdbid, chainID, ligand_atoms, axes, pos, ligand  = data
            print(f'{pdbid},   {chainID},   {pos}')
        # if len(ligand_atoms) == 1:
            # # if pdbid == '5O0Z':
            # #     continue
            # pdbid = '6FJJ'
            # chainID = 'A'
            # pos = '302'
            # ligand = 'LSA'
            #     print(i)
            #     continue
            # print('=='*50)
            print(pdbid, chainID, pos, ligand)
            file_name = 'pdb_files_all_new/' + pdbid.lower() + '.pdb'
            result = check_models(file_name)
            if result:
                model_index = get_model_index(file_name, ligand, chainID, pos, result)
                pdb_atoms4chainIDs = read_pdb(file_name, model_index, result)
            else:
                pdb_atoms4chainIDs = read_pdb(file_name, result)
            if pdb_atoms4chainIDs:
                min_dist_chain = {}
                for chain, atoms_data in pdb_atoms4chainIDs.items():
                    atomTypes, atomCoords = atoms_data
                    dist = pairwise_distances(axes, np.array(atomCoords).T[1:].T)
                    min_dist_chain[chain] = np.min(dist)
                # closest_chain = min(min_dist_chain, key=min_dist_chain.get)
                min_dist_chain_non_zero = {key: value for key, value in min_dist_chain.items() if value > 0}

                closest_chain = min(min_dist_chain_non_zero, key=lambda dict_key: min_dist_chain_non_zero[dict_key])

                if min_dist_chain[closest_chain] == 0:
                    raise Exception('check min function !!!')
                if closest_chain != chainID:
                    closest_chain_inconsistent_list.append({'pdbid': pdbid, 'chainID': chainID, 'pos': pos, 'ligand': ligand, 'closest_chain': closest_chain})
                    # print(pdbid, chainID, pos, ligand)
                else:
                    assert closest_chain == chainID
                atomTypes_final, atomCoords_final = pdb_atoms4chainIDs[closest_chain]
                try:
                    assert len(atomCoords_final) > 206
                    distance = pairwise_distances(axes, np.array(atomCoords_final).T[1:].T)
                    index_protein_atoms = list(set(get_indices_of_k_smallest(distance, 206)))
                    atomCoords_final_env = [atomCoords_final[index] for index in index_protein_atoms]
                    data_deque.append([pdbid, chainID, ligand_atoms, axes, pos, ligand, atomCoords_final_env])
                    # data.append(atomCoords_final_env)      # {'train' : [[pdbid, chainID, ligand_atoms, axes, pos, ligand, atomtypes, atomCoords], [pdbid, chainID, ligand_atoms, axes, pos, ligand, atomtypes, atomCoords]]}
                except AssertionError:
                    assertError_file_list.append([pdbid, chainID, pos, ligand, len(atomCoords_final)])
                    print('AssertionError raised: pdb id: ', pdbid, 'chain id: ', chainID, 'pos: ', pos, 'ligand: ', ligand)
            else:
                pdb_atoms4chainIDs_empty_data_list.append([pdbid, chainID, pos, ligand])
                print('======= empty pdb_atoms4chainIDs', 'pdbid', 'chainID', 'pos', 'ligand', '=======')
        # dataset_env_atoms[f'{key}_ligand_env_atoms'] = data_deque

        f = open(f"{key}_env_atoms_deque.pickle","wb")                   # all the required data
        pickle.dump(data_deque,f)
        f.close()

    f = open("pdb_atoms4chainIDs_empty_data_list.pickle","wb")      # something wrong in reading protein atoms
    pickle.dump(pdb_atoms4chainIDs_empty_data_list,f)
    f.close()


    f = open("assertError_file_list.pickle","wb")      # pdb file does not have  206 protein atoms
    pickle.dump(assertError_file_list,f)
    f.close()


    import json
    with open('closest_chain_inconsistent_list.json', 'w') as file:
        for ele in closest_chain_inconsistent_list:
            file.write(json.dumps(ele, indent = 4) + '\n')





