import sys

from .base_model import BaseModel
from . import networks
from . import rmsd_utils
import numpy as np
import torch



class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
        #                               opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        # print(input)
        self.ligand_atoms = input['key']
        _, _, _, self.ligand, _, _, self.atoms = self.ligand_atoms
        # ligandLen = len(self.ligand)

        center_loc = np.squeeze([i.numpy() for i in self.ligand[0]])
        start_center_loc = self.path_generator(self.atoms, center_loc.tolist())
        starting_ligand = np.array([22] + np.array(start_center_loc).T.tolist()).T
        # starting_ligand = np.vstack([[22] + np.array(start_center_loc).T.tolist()]).T

        self.real_cpu_dis_matrix, \
        self.target_dis_matrix, \
        self.starting_rmsd, \
        self.random_env_atoms_data = rmsd_utils.get_input_dis_matrix(starting_ligand, self.atoms, self.ligand)

        self.real_dis_matrix = self.real_cpu_dis_matrix.to(self.device)
        self.image_paths = input['A_paths']
        self.ligandLength = 1
        self.atomsLength = len(self.atoms)

    def get_input_second(self):

        sub_pred_dist_matrix = self.get_original_matrix(self.pred_dis_matrix[0, 0, :, :], self.ligandLength, self.atomsLength)

        pred_dm = ((sub_pred_dist_matrix + 1) / 2) * 80  # remove normalization and extract the distance matrix
        # print("ligand length is: ", ligandLength)
        pred_locs = rmsd_utils.get_locs(pred_dm, self.random_env_atoms_data, self.ligandLength)

        assert pred_locs.shape == (self.ligandLength, 4)

        self.real_cpu_dis_matrix2, \
        self.target_dis_matrix2, \
        _, \
        _ = rmsd_utils.get_input_dis_matrix(pred_locs, self.atoms, self.ligand, permute=False)
        assert self.target_dis_matrix2.all() ==  self.target_dis_matrix.all()
        self.real_dis_matrix_2step = self.real_cpu_dis_matrix2.to(self.device)



    def forward(self, iterations):
        """Run forward pass."""

        if iterations == 0:
            self.pred_dis_matrix = self.netG(self.real_dis_matrix)
        if iterations == 1:
            self.pred_dis_matrix = self.netG(self.real_dis_matrix)
            self.get_input_second()
            self.pred_dis_matrix_2step = self.netG(self.real_dis_matrix_2step)


    def optimize_parameters(self):
        """No optimization for test model."""
        pass


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
        verts_loc = np.array([np.stack(atoms_locations[i][1:]).reshape(3,) for i in range(len(atoms_locations))])
        # verts_loc = np.array(atoms_locations).T[1:].T
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