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

import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from models.Preprocessing import save, load
import wandb
import sys
import numpy as np

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

    if opt.eval:
        model.eval()
    # ligand_atoms_pair = getattr(dataset, 'ligand_atoms_pair')
    # selected_ligand_atom_pair_data = getattr(dataset, 'selected_ligand_atom_pair')

    for i, data in enumerate(dataset):
        print(i, data)
        model.set_input(data)  # unpack data from data loader
        model.test(opt.iterations)         # run inference
        if opt.iterations > 0:
            pred_ligand_rmsd_1step, pred_ligand_rmsd_2step, pred_starting_rmsd = model.compute_rmsd_2step()
            ligand_rmsd.append(pred_ligand_rmsd_1step)
            ligand_rmsd_2step.append(pred_ligand_rmsd_2step)
        if opt.iterations == 0:
            pred_ligand_rmsd, pred_starting_rmsd = model.compute_rmsd()
            ligand_rmsd.append(pred_ligand_rmsd)
        starting_rmsd.append(pred_starting_rmsd)

        # visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
        if i % 1000 == 0:  # save images to an HTML file
        # if i == 10 :  # save images to an HTML file
            print('processing (%04d)-th ligands...' % (i))

        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    # webpage.save()  # save the HTML
    print('--'*100)
    print(f'current epoch is: {opt.epoch}')
    print(f'mean is : {np.mean(ligand_rmsd)}, median is {np.median(ligand_rmsd)}, std is {np.std(ligand_rmsd)}')
    print('--' * 100)


    save(ligand_rmsd, f'./output/ligand_rmsd_{opt.epoch}_seed{opt.seed}_iter{opt.iterations}.pickle')
    save(ligand_rmsd_2step, f'./output/ligand_rmsd_2step_{opt.epoch}_seed{opt.seed}_iter{opt.iterations}.pickle')
   # save(dists_S_rmsd, f'./dists_S_rmsd_{opt.epoch}_try1.pickle')
    save(starting_rmsd, f'./output/starting_rmsd_{opt.epoch}_seed{opt.seed}_iter{opt.iterations}.pickle')

print('All is done')