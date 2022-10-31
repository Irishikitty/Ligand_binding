# ----------------------------------------------------------------------
# This script generate tensor of size [23, 250, 250] from 3D coordinates
# ----------------------------------------------------------------------

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.metrics import pairwise_distances
    import pickle
    import warnings
    import random
    warnings.filterwarnings('ignore') # suppress overflow warning in exp
    TRAINING = 'train'


    def data_toTensor(ligand, atoms, start = False, dim_max=256, channels = 23):
        '''
        3D coordinates of input (ligand + atom)
        Ouptut tensor with shape [23, dim_sum = 256, 256]

        Input:
            :param ligand: list of [22 x y z]
            :param atoms: list of [c x y z]
            :param dim_max: 250 as default
        Output:

        '''
        if start == False:
            numAtoms, _ = np.array(atoms).shape
            numLigandAtoms, _ = np.array(ligand).shape

            A_coordinates = pairwise_distances([[x, y, z] for [_, x, y, z] in ligand] + [[x, y, z] for [i, x, y, z] in atoms])
            types = [i for [i, _, _, _] in atoms]
            container = torch.zeros(channels-1, *A_coordinates.shape)
            for i, c in enumerate(types):
                for j in range(len(types)):
                    container[c - 1, i, j] += 1
                    container[c - 1, j, i] += 1
            container[container > 0] = 1  # as indicator
            # build tensor
            A_coordinates = torch.from_numpy(A_coordinates.reshape(-1, *A_coordinates.shape))
            A_coordinates = torch.cat([A_coordinates, container], dim=0)
            A_coordinates[22, :numLigandAtoms, :numLigandAtoms] = 1
            # move blocks
            out = torch.zeros(channels, dim_max, dim_max)  # first channel container
            out[:, :numLigandAtoms, :numLigandAtoms] += A_coordinates[:, :numLigandAtoms, :numLigandAtoms]  # upper left
            out[:, 50:50 + numAtoms, 50:50 + numAtoms] += A_coordinates[:, numLigandAtoms:, numLigandAtoms:]  # lower right
            out[:, :numLigandAtoms, 50:50 + numAtoms] += A_coordinates[:, :numLigandAtoms, numLigandAtoms:]  # rectangle upper
            out[:, 50:50 + numAtoms, :numLigandAtoms] += A_coordinates[:, numLigandAtoms:, :numLigandAtoms]  # rectangle lower
            return out

        if start == True:
            '''
            Fill out upper right and lower left blocks with 0
            '''
            numAtoms, _ = np.array(atoms).shape
            numLigandAtoms, _ = np.array(ligand).shape

            A_coordinates = pairwise_distances(
                [[x, y, z] for [_, x, y, z] in ligand] + [[x, y, z] for [i, x, y, z] in atoms])
            types = [i for [i, _, _, _] in atoms]
            container = torch.zeros(channels - 1, *A_coordinates.shape)
            for i, c in enumerate(types):
                for j in range(len(types)):
                    container[c - 1, i, j] += 1
                    container[c - 1, j, i] += 1
            container[container > 0] = 1  # as indicator
            # build tensor
            A_coordinates = torch.from_numpy(A_coordinates.reshape(-1, *A_coordinates.shape))
            A_coordinates = torch.cat([A_coordinates, container], dim=0)
            A_coordinates[22, :numLigandAtoms, :numLigandAtoms] = 1
            # move blocks
            out = torch.zeros(channels, dim_max, dim_max)  # first channel container
            out[:, :numLigandAtoms, :numLigandAtoms] += A_coordinates[:, :numLigandAtoms, :numLigandAtoms]  # upper left
            out[:, 50:50 + numAtoms, 50:50 + numAtoms] += A_coordinates[:, numLigandAtoms:, numLigandAtoms:]  # lower right
            out1 = out.clone()
            out[:, :numLigandAtoms, 50:50 + numAtoms] += A_coordinates[:, :numLigandAtoms,
                                                         numLigandAtoms:]  # rectangle upper
            out[:, 50:50 + numAtoms, :numLigandAtoms] += A_coordinates[:, numLigandAtoms:,
                                                         :numLigandAtoms]  # rectangle lower
            return out1, out

    def data_to_0_1(sample):
        '''
        Normalize to [-1, 1]
        :param sample: input mx/tensor
        :return: normalized input
        '''

        out = 2 * sample / 80 - 1
        return out

    def distance_3D(x, y, axis=None):
        diff = np.array(x) - np.array(y)
        diff = diff ** 2
        return np.sqrt(np.sum(diff, axis=axis))
    def distance_to_atoms(candidate, verts, axis=None):
        return distance_3D(candidate, verts, axis=axis)
    def path_generator(atoms_locations, ligand_starting_loc):

        next_loc = None
        verts_loc = np.array(atoms_locations).T[1:].T
        current_loc = ligand_starting_loc
        NumberofPoints = np.random.randint(1, 100)

        for step_index in range(1, NumberofPoints + 1):

            candidate_d_dict = {}
            T = 700 / (step_index + 1)
            direction_list = ['right', 'left', 'forward', 'backward', 'up', 'down']
            denominator_ss_list = []
            for direction in direction_list:
                # step with randomness
                candidate_loc = generate_the_next_candidate_point(current_loc, direction)
                candidate_d_dict[direction] = candidate_loc
                distance = distance_to_atoms(list(candidate_loc), verts_loc, axis=1)
                ss = np.sum(np.asarray(distance) ** 2)
                denominator_ss_list.append(ss / T)

            p_list = []
            for a in denominator_ss_list:
                diff = np.array(denominator_ss_list) - a
                temp = np.sum(np.exp(np.array(diff)))
                p_list.append(1 / temp)

            direction = np.random.choice(direction_list, p=p_list)
            next_loc = candidate_d_dict[direction]

            current_loc = list(next_loc)

        return next_loc

    def generate_the_next_candidate_point(current_timestep_point: list, dir: str) -> tuple:
        next_candidate = np.array(current_timestep_point)
        increment = 0.5
        mu, sigma = 0, 0.1

        if dir == 'right':
            next_candidate[0] = next_candidate[0] + increment + np.random.normal(mu, sigma, 1)
        if dir == 'left':
            next_candidate[0] = next_candidate[0] - increment + np.random.normal(mu, sigma, 1)
        if dir == 'forward':
            next_candidate[1] = next_candidate[1] - increment + np.random.normal(mu, sigma, 1)
        if dir == 'backward':
            next_candidate[1] = next_candidate[1] + increment + np.random.normal(mu, sigma, 1)
        if dir == 'up':
            next_candidate[2] = next_candidate[2] + increment + np.random.normal(mu, sigma, 1)
        if dir == 'down':
            next_candidate[2] = next_candidate[2] - increment + np.random.normal(mu, sigma, 1)

        return tuple(next_candidate)

    SEED = 10
    np.random.seed(SEED)
    random.seed(SEED)

    df_training = pd.read_pickle('all_dataset_ligand_env.pickle')
    df_training = df_training[TRAINING]

    num_samples = len(df_training)
    pdbs = []
    chains = []
    ligand_names = []
    initial_locations = []
    f = open("all_dataset_initial.txt", "a")
    for i in range(num_samples):
        try:
            pdbid, chain, _, ligand_loc_list, _, ligand_name, atoms = df_training[i]
            center_loc = np.mean(np.array(ligand_loc_list), axis=0)
            start_center_loc = path_generator(atoms, center_loc.tolist())
            starting_ligand_loc = np.array(ligand_loc_list) + np.array(start_center_loc) - center_loc

            f.write(pdbid+';'+chain+';'+ligand_name+';'+str(starting_ligand_loc[0]))
            pdbs.append(pdbid)
            chains.append(chain)
            ligand_names.append(ligand_name)
            initial_locations.append(starting_ligand_loc)

            if (i % 10) == 0:
                print(f'Current sample is {i}-th \n')
        except:
            print(i)

    f.close()

    with open("pdbs.txt", "wb") as fp:
        pickle.dump(pdbs, fp)
    with open("chains.txt", "wb") as fp:
        pickle.dump(chains, fp)
    with open("ligand_names.txt", "wb") as fp:
        pickle.dump(ligand_names, fp)
    with open("initial_locations.txt", "wb") as fp:
        pickle.dump(initial_locations, fp)

    print('finished!')