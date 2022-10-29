import copy
import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from . import rmsd_utils
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class BaseModel(ABC):

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        # if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self, iterations):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self, iterations):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward(iterations)
            # self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def compute_rmsd(self):

        pred_dis_matrix = getattr(self, 'pred_dis_matrix').detach().clone()

        random_atom_data = getattr(self, 'random_env_atoms_data')
        starting_dis_rmsd = getattr(self, 'starting_rmsd')
        ligandLength = getattr(self, 'ligandLength')
        atomsLength = getattr(self, 'atomsLength')
        true_ligands_atoms = getattr(self, 'ligand_atoms')
        _, _, _, true_ligands, _, _, atoms = true_ligands_atoms

        assert len(pred_dis_matrix.shape) == 4
        sub_pred_dist_matrix = self.get_original_matrix(pred_dis_matrix.cpu()[0, 0,:, :], ligandLength, atomsLength)

        pred_dm = ((sub_pred_dist_matrix + 1) / 2) * 80  # remove normalization and extract the distance matrix
        # print("ligand length is: ", ligandLength)
        pred_rmsd, predicted_loc = rmsd_utils.get_rmsd(pred_dm, random_atom_data, true_ligands, ligandLength)

        return pred_rmsd, starting_dis_rmsd, predicted_loc, pred_dm

    def compute_rmsd_2step(self):

        pred_dis_matrix = getattr(self, 'pred_dis_matrix').detach().clone()
        pred_dis_matrix_2step = getattr(self, 'pred_dis_matrix_2step').detach().clone()

        random_atom_data = getattr(self, 'random_env_atoms_data')
        starting_dis_rmsd = getattr(self, 'starting_rmsd')
        ligandLength = getattr(self, 'ligandLength')
        atomsLength = getattr(self, 'atomsLength')
        true_ligands_atoms = getattr(self, 'ligand_atoms')
        _, _, true_ligands, atoms = true_ligands_atoms


        assert len(pred_dis_matrix.shape) == 4
        sub_pred_dist_matrix = self.get_original_matrix(pred_dis_matrix[0, 0,:, :], ligandLength, atomsLength)
        sub_pred_dist_matrix_2step = self.get_original_matrix(pred_dis_matrix_2step[0, 0,:, :], ligandLength, atomsLength)

        pred_dm = ((sub_pred_dist_matrix + 1) / 2) * 80  # remove normalization and extract the distance matrix
        pred_dm_2step = ((sub_pred_dist_matrix_2step + 1) / 2) * 80  # remove normalization and extract the distance matrix
        # print("ligand length is: ", ligandLength)
        pred_rmsd = rmsd_utils.get_rmsd(pred_dm, random_atom_data, true_ligands, ligandLength)
        pred_rmsd_2step = rmsd_utils.get_rmsd(pred_dm_2step, random_atom_data, true_ligands, ligandLength)

        return pred_rmsd, pred_rmsd_2step, starting_dis_rmsd


    def get_original_matrix(self, temp_input_matrix, ligandLength, atomsLength, avg = True):
        '''
        Recover original matrix [protein_env + ligand, protein_env + ligand]

        :param temp_input_matrix: predicted output
        :param ligandLength:
        :param atomsLength:
        :param avg: if True, then take the avg of two bars
        :return:
        '''
        input_matrix = copy.deepcopy(temp_input_matrix)
        assert input_matrix.shape == (256, 256)

        side = ligandLength + atomsLength
        sub_input_matrix = np.zeros((side,side))

        # protein to the upper left; ligand to lower right
        sub_input_matrix[:atomsLength, :atomsLength] += np.array(temp_input_matrix[50:50+atomsLength, 50:50+atomsLength])  # protein
        sub_input_matrix[200:200+ligandLength, 200:200+ligandLength] += np.array(temp_input_matrix[:ligandLength, :ligandLength])  # ligand

        if avg == True:
            avgg = temp_input_matrix[:ligandLength, 50:50+atomsLength].T + temp_input_matrix[50:50 + atomsLength, :ligandLength]
            sub_input_matrix[:atomsLength,200:] += np.array(avgg/2)
        else:
            sub_input_matrix[200:, :atomsLength] += np.array(temp_input_matrix[:ligandLength, 50:50+atomsLength])
            sub_input_matrix[:atomsLength,200:] += np.array(temp_input_matrix[50:50 + atomsLength, :ligandLength])

        return sub_input_matrix


    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

