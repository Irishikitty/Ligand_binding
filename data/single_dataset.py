from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import pandas as pd
import pickle
from .Preprocessing import data_preprocessing_24channel_multi_distill
import numpy as np
import torch.nn as nn


# ========================================================================================


def save(a, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(a, handle, protocol=4)

# ========================================================================================

def load(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

# ========================================================================================

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        # input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))
        #

        self.dir =  opt.dataroot
        print("loading: ", self.dir)
        self.max_dims = 256
        self.ligand_atoms_pair = pd.read_pickle(self.dir  + '/all_dataset_ligand_env.pickle')['validation']
        self.len = len(self.ligand_atoms_pair)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        # A = self.transform(A_img)

        # A = self.ligand_loc_key[index]

        key_path = self.dir
        return {'key': self.ligand_atoms_pair[index], 'A_paths': key_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len


