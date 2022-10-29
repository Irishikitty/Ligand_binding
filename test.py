"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from sklearn.metrics import pairwise_distances

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.Preprocessing import save, load
import wandb
import numpy as np
from models.rmsd_utils import get_neighbor_points, coherent_point_registration, intersection,compute_coord_fitness
import json
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

# model_name, data_name, output_name
NAME = './output/'

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    seed_everything(opt.seed)
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given  and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)              # regular setup: load and print networks; create schedulers

    print(opt)
    # print('Here'*100)
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    """

    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    """
    ligand_rmsd = []
    ligand_rmsd_2step = []
    dists_S_rmsd = []
    starting_rmsd = []
    predicted_coords = []
    conformation_rmsds = []
    coord_fitnesss = []
    pdbfiles_s = []
    distances = []

    if opt.eval:
        model.eval()
    # ligand_atoms_pair = getattr(dataset, 'ligand_atoms_pair')
    # selected_ligand_atom_pair_data = getattr(dataset, 'selected_ligand_atom_pair')

    for i, data in enumerate(dataset):
        # print(i, data)
        try:
            model.set_input(data)  # unpack data from data loader
            model.test(opt.iterations)         # run inference
            if opt.iterations > 0:
                pred_ligand_rmsd_1step, pred_ligand_rmsd_2step, pred_starting_rmsd, pred_dm = model.compute_rmsd_2step()
                ligand_rmsd.append(pred_ligand_rmsd_1step)
                ligand_rmsd_2step.append(pred_ligand_rmsd_2step)
            if opt.iterations == 0:
                pred_ligand_rmsd, pred_starting_rmsd, predicted_loc, pred_dm = model.compute_rmsd()
                ligand_rmsd.append(pred_ligand_rmsd)
            starting_rmsd.append(pred_starting_rmsd)
            ligand_len = len(data['key'][2])
        except:
            print(i)



        # Compute conformation RMSD -----------------------        # true_loc = [np.stack(data['key'][2][i]).reshape(4,) for i in range(len(data['key'][2]))][0][1:]
        # true_loc = np.squeeze([i.numpy() for i in data['key'][3][0]])
        # predicted_loc = predicted_loc[0][1:]
        #
        # if ligand_len == 1:
        #     try:
        #         # Compute conformation and directionality -----------
        #         PDBname = opt.dataroot +'/pdbdata/' + data['key'][0][0].lower()
        #         coord_rmsd, coord_fitness, third_distance = compute_coord_fitness(predicted_loc, true_loc, PDBname, opt.radius)
        #         distances.append(third_distance)
        #         conformation_rmsds.append(coord_rmsd)
        #         coord_fitnesss.append(coord_fitness)
        #         print(f'coord_rmsd: {np.round(coord_rmsd,2)}     coord_fitness:  {np.round(coord_fitness,2)}     distance {np.round(third_distance,2)}')
        #     except:
        #         pdbfiles_s.append(data['key'][0][0].lower())
        #         continue
            # Drawing plots -----------------------------------
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.scatter3D(pt1[:, 0], pt1[:, 1], pt1[:, 2], facecolor='red', s=25, alpha=1)
            # ax.scatter3D(pt2[:, 0], pt2[:, 1], pt2[:, 2], facecolor='blue', s=25, alpha=1)
            # ax.scatter3D(pt3[:, 0], pt3[:, 1], pt3[:, 2], facecolor='purple', s=25, alpha=1)
            # ax.scatter3D(predicted_loc[0][1:][0],predicted_loc[0][1:][1],predicted_loc[0][1:][2], facecolor='gray', s=15*5)
            # ax.scatter3D(true_loc[0], true_loc[1], true_loc[2], facecolor='black', s=15*5)
            # ax.set_title(f'Distance: {np.round(np.linalg.norm(predicted_loc-true_loc,2),2)}   RMSD: {np.round(conformation_rmsd,2)}')
            # plt.savefig(f'./output/output_pos/images/img_{i}.png')
            # plt.show()

        if i % 1000 == 0:  # save images to an HTML file
        # if i == 10 :  # save images to an HTML file
            print('processing (%04d)-th ligands...' % (i))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    # webpage.save()  # save the HTML
    print('--'*100)
    print(f'current epoch is: {opt.epoch}')
    # print(f'mean is : {np.mean(ligand_rmsd)}, median is {np.median(ligand_rmsd)}, std is {np.std(ligand_rmsd)}')
    print(f'The mean coord_rmsd: {np.mean(conformation_rmsds), np.mean(np.array(conformation_rmsds)[np.array(conformation_rmsds)>0])}')
    print(f'The mean coord_fit: {np.mean(coord_fitnesss), np.mean(np.array(coord_fitnesss)[np.array(conformation_rmsds) > 0])}')
    print('--' * 100)

    import pickle
    with open(NAME+'opt.pickle', 'wb') as handle:
        pickle.dump(vars(opt), handle, protocol=pickle.HIGHEST_PROTOCOL)

    save(ligand_rmsd, NAME + '/ligand_rmsd.pickle')
    # save(starting_rmsd, NAME + '/starting_rmsd.pickle')
    # save(predicted_coords, NAME + '/pred_true.pickle')
    # save(ligand_len, NAME +'/ligandlen.pickle')
    save(pdbfiles_s, NAME + '/pdbfiles_s.pickle')
    save(conformation_rmsds, NAME + '/conformation_rmsd.pickle')
    save(coord_fitnesss, NAME + '/coord_fitness.pickle')

print('All is done')