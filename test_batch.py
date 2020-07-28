"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
# from utils import get_config, get_data_loader_folder_for_test, pytorch03_to_pytorch04, load_inception
from utils import get_config, get_data_loader_folder_for_test, pytorch03_to_pytorch04
# from trainer import MUNIT_Trainer, UNIT_Trainer
from trainer import MUNIT_Trainer
from torch import nn
# from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import sys
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
parser.add_argument('--inception_a', type=str, default='.',
                    help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='.',
                    help="path to the pretrained inception network for domain B")

### test limited labels
parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
parser.add_argument('--config', default='configs/mnist2svhn_002_infoStyle.yaml', type=str, help="net configuration")
parser.add_argument('--output_only', default=True, help="whether only save the output images or also save the input images")

# label = 9
parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainB/'+str(label), type=str)

# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/MUNIT-reremake/Models/MUNIT_CC6lU_1shot/outputs/mnist2svhn_002_infoStyle/gen_00400000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6lU_LL0/ckpt_400k/A10_BFullX2_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--num_style', type=int, default=2, help="number of styles to sample")
label = 0
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00080000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainA_1k+5k/9', type=str, help="output image path")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/ckpt_370k/trainA_1k_5k/'+str(label), type=str, help="output image path")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC_6losses/checkpoints/gen_00350000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6losses/ckpt_350k/trainA_60k_5k/'+str(label), type=str, help="output image path")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC_6losses/checkpoints/gen_00350000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6losses/ckpt_350k/trainA_60k_10k/'+str(label), type=str, help="output image path")

# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_0_5k/'+str(label), type=str, help="output image path")

# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_1k_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainB/'+str(label), type=str)

# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_10k_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainB/'+str(label), type=str)

# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_20k_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainB/'+str(label), type=str)
# LIMIT_SAMPLE = 2000

# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_40k_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainB/'+str(label), type=str)

# LIMIT_SAMPLE = 4000
# parser.add_argument('--num_style', type=int, default=1, help="number of styles to sample")
parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainB/'+str(label), type=str)

# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_max_generated/'+str(label), type=str, help="output image path")

# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL1k_downloaded/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL1k/ckpt_370k/trainA_maxX3_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--num_style', type=int, default=1, help="number of styles to sample")

### MUNIT_CC6l_LL0/gen_00370000
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_LL0/gen_00370000.pt', type=str, help="checkpoint of autoencoders")

# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL0/ckpt_370k/trainA_0_B10k/'+str(label), type=str, help="output image path")
# LIMIT_SAMPLE = 1000

# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL0/ckpt_370k/trainA_0_B40k/'+str(label), type=str, help="output image path")
# LIMIT_SAMPLE = 4000

# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL0/ckpt_370k/trainA_0_Bmax/'+str(label), type=str, help="output image path")
# LIMIT_SAMPLE = None

# # CC6l A 0 B full x2
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL0/ckpt_370k/trainA_0_Bfull_x2_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--num_style', type=int, default=2, help="number of styles to sample")
# LIMIT_SAMPLE = None

# CC6l A 0 B full x4
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_LL0/ckpt_370k/trainA_0_Bfull_x4_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--num_style', type=int, default=4, help="number of styles to sample")
# LIMIT_SAMPLE = None

##### MUNIT_CC6lU_1shot / gen_00400000
LIMIT_SAMPLE = None
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6lU_1shot/gen_00400000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6lU_LL0/ckpt_400k/A10_BFull_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6lU_1shot/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6lU_LL0/ckpt_370k/A10_BFull_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6lU_1shot/gen_00300000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6lU_LL0/ckpt_300k/A10_BFull_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6lU_1shot/gen_00200000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6lU_LL0/ckpt_200k/A10_BFull_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6lU_1shot/gen_00500000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6lU_LL0/ckpt_500k/A10_BFull_generated/'+str(label), type=str, help="output image path")
parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6lU_1shot/gen_00600000.pt', type=str, help="checkpoint of autoencoders")
parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6lU_LL0/ckpt_600k/A10_BFull_generated/'+str(label), type=str, help="output image path")
parser.add_argument('--num_style', type=int, default=1, help="number of styles to sample")

# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6l_1shot_cp_fr_CC4l_1shot/gen_00400000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6l_1shot_cp_fr_CC4l_1shot/ckpt_400k/A10_BFull_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--num_style', type=int, default=1, help="number of styles to sample")


# ### Compare CC4 vs CC6: MUNIT_CC4l_LL1k --> Conclusion: same for 1k labels!
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC4l_LL1k/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC4l_LL1k/ckpt370k/trainA_1k_B10k_generated/'+str(label), type=str, help="output image path")
# LIMIT_SAMPLE = 1000

### CC4 one shot
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC4l_1shot/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC4l_1shot/ckpt370k/trainA_10_B_full_generated/'+str(label), type=str, help="output image path")
# LIMIT_SAMPLE = None

# ### CC4 one shot B_full x2
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC4l_1shot/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC4l_1shot/ckpt370k/trainA_10_B_full_x2_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--num_style', type=int, default=2, help="number of styles to sample")
# LIMIT_SAMPLE = None

# ### CC4 one shot B_full x4
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC4l_1shot/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC4l_1shot/ckpt370k/trainA_10_B_full_x4_generated/'+str(label), type=str, help="output image path")
# parser.add_argument('--num_style', type=int, default=4, help="number of styles to sample")
# LIMIT_SAMPLE = None

parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/MUNIT-reremake/Models/MUNIT_CC6lU_1shot_dup/outputs/mnist2svhn_002_infoStyle/checkpoints/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/augmented_by_infoMUNIT/MUNIT_CC6lU_1shot_dup/ckpt_370k/A10_BFull_generated/'+str(label), type=str, help="output image path")
parser.add_argument('--num_style', type=int, default=1, help="number of styles to sample")

LIMIT_SAMPLE = None

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder_for_test(opts.input_folder, 1, False, new_size=None, crop=False)

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
# elif opts.trainer == 'UNIT':
#     trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode  # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode  # decode function

if opts.trainer == 'MUNIT':
    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if LIMIT_SAMPLE is not None:
            if i >= LIMIT_SAMPLE:
                break
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)
        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(),
                                                               volatile=True)
        # print(style)

        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            # path = os.path.join(opts.output_folder+"_%02d"%j,basename)
            path = os.path.join(opts.output_folder, f'{basename[:-4]}_{j}{basename[-4:]}')
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0,
                              normalize=True)
