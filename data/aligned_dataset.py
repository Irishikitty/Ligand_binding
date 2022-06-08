import os
import random

from .base_dataset import BaseDataset, get_params, get_transform
import pickle as pickle
import torch
from .Preprocessing import data_preprocessing_2channel, seed_everything, data_preprocessing_24channel_multi, data_preprocessing_52channel, data_preprocessing_30channel, \
    data_preprocessing_24channel_multi_distill
import numpy as np
import torch.nn as nn

def save(a, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(a, handle, protocol=4)

# ========================================================================================

def load(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

# ========================================================================================

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    # def __init__(self, opt):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            size: box size
        """
        # BaseDataset.__init__(self, opt)
        BaseDataset.__init__(self, opt)

        self.dir =  opt.dataroot
        print("loading from ", self.dir)
        self.max_dims = 250
        self.ligand_atoms_pair = load(self.dir  + '/train_ligand_env_coords.pickle')
        self.len = len(self.ligand_atoms_pair)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        pdbid, ligand_name, ligand, atoms= self.ligand_atoms_pair[index]
        ligand_loc_list = [ele[1:] for ele in ligand]
        center_loc = np.mean(np.array(ligand_loc_list), axis=0)
        assert center_loc.shape == (3,)
        start_center_loc = self.path_generator(atoms, center_loc.tolist())
        starting_ligand_loc = np.array(ligand_loc_list) + np.array(start_center_loc) - center_loc
        starting_ligand = np.vstack([[22]*len(ligand), starting_ligand_loc.T]).T

        tempFakeA, tempTrueB = data_preprocessing_24channel_multi_distill(starting_ligand, atoms, ligand)

        assert tempFakeA.shape[2] <= self.max_dims
        assert tempTrueB.shape[2] <= self.max_dims

        if tempFakeA.shape[2] % 2 == 0:
            padding = self.max_dims - tempFakeA.shape[1]
            m = nn.ZeroPad2d(padding // 2)
        else:
            assert tempFakeA.shape[2] % 2 == 1
            padding = self.max_dims - tempFakeA.shape[1]
            m = nn.ZeroPad2d((padding//2, padding//2+1, padding//2, padding//2+1))

        A, B = m(torch.tensor(tempFakeA)), m(torch.tensor(tempTrueB))
        A[0, :, :] = (A[0, :, :] / 82) * 2 - 1
        B[0, :, :] = (B[0, :, :] / 82) * 2 - 1

        assert A.shape == (24, 250, 250)
        assert B.shape == (24, 250, 250)

        return {'A': A.float(), 'B': B.float(), 'A_paths': "None", 'B_paths': "None",
                'Ligand_length': len(ligand) , 'env_atoms_length': len(atoms) }


    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len

    # ==============================================================================

    def generate_the_next_candidate_point(self, current_timestep_point: list, dir: str) -> tuple:
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

    # ==============================================================================


    def distance_3D(self, x, y, axis=None):
        diff = np.array(x) - np.array(y)
        diff = diff ** 2
        return np.sqrt(np.sum(diff, axis=axis))

    # ==============================================================================

    def distance_to_atoms(self, candidate, verts, axis=None):
        # return tuple(distance_3D(candidate, verts, axis=axis))
        return self.distance_3D(candidate, verts, axis=axis)

    # ==============================================================================

    def softmax(self, x):
        """ applies softmax to an input x"""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    # ========================================================================================


    def path_generator(self, atoms_locations, ligand_starting_loc):

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
                candidate_loc = self.generate_the_next_candidate_point(current_loc, direction)
                candidate_d_dict[direction] = candidate_loc
                distance = self.distance_to_atoms(list(candidate_loc), verts_loc, axis=1)
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


if __name__=="__main__":

    def seed_everything(seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    seed_everything(42)
    dir = "../datasets"
    dataset = AlignedDataset(dir)
    #
    # current_data = data[5722]
    # print(data[5722])
    # A = current_data['A']
    # B = current_data['B']
    # print(A)
    # print(B)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False)
    # min = 9999
    # data_dict = next(iter(dataloader))
    # aa = data_dict['A']
    # print(aa[0].shape)
    # print(aa[0])
    # print(torch.max(aa[0]))



    for index, data_dict in enumerate(dataloader):
        print("**"*100)
        print("current index is:", index)
        print(data_dict['A'].shape)
        print(data_dict['B'].shape)
        print(data_dict['Ligand_length'])
        print(data_dict['env_atoms_length'])
        print(torch.max(data_dict['A']))
        # if min >= torch.max(data_dict['A']):
        #     min = torch.max(data_dict['A'])
        # print('current min dist is :', min)



    #
    # a = next(iter(dataloader))
    # #
    # print(torch.max(a['A']))