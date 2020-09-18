"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder_test_batch, pytorch03_to_pytorch04, load_inception
from trainer import MUNIT_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os


parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
# parser.add_argument('--input_folder', type=str, help="input image folder")
# parser.add_argument('--output_folder', type=str, help="output image folder")
# parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--seed', type=int, default=1, help="random seed")
# parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
# parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
# parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
# parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
# parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
# parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
# parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
# parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
# parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")

##################### Test batch for workshop version: following paper https://arxiv.org/pdf/1804.04732.pdf #####################
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=100, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', default=True, action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--compute_IS', default='true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', default='true', help="whether to compute Conditional Inception Score or not")
# parser.add_argument('--inception_a', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/inception_v3_google-1a9a5a14.pth', help="path to the pretrained inception network for domain A")
# parser.add_argument('--inception_b', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/inception_v3_google-1a9a5a14.pth', help="path to the pretrained inception network for domain B")
parser.add_argument('--inception_a', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/inception_v3_google-1a9a5a14.pth', help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/inception_v3_google-1a9a5a14.pth', help="path to the pretrained inception network for domain B")
LIMIT_INPUT = 200

# ##### test batch for experience 006_MUNIT_origin_edge2shoe_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/006_MUNIT_origin_edge2shoe_64/outputs/edges2shoes_folder/checkpoints/gen_00800000.pt', type=str, help="checkpoint of autoencoders")

# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2shoes/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/006_MUNIT_origin_edge2shoe_64/tests/test_batch_200i/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2shoes/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/006_MUNIT_origin_edge2shoe_64/tests/test_batch_200i/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)



# ##### test batch for experience 005_MUNIT_origin_edge2bag_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/005_MUNIT_origin_edge2bag_64/outputs/edges2handbags_folder/checkpoints/gen_00800000.pt', type=str, help="checkpoint of autoencoders")

# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/005_MUNIT_origin_edge2bag_64/tests/test_batch_200i/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/005_MUNIT_origin_edge2bag_64/tests/test_batch_200i/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# ##### test batch for experience 004_edge2shoe_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/004_edge2shoe_64/outputs/edges2shoes_folder/checkpoints/gen_00800000.pt', type=str, help="checkpoint of autoencoders") 

# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2shoes/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/004_edge2shoe_64/tests/test_batch_200i/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2shoes/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/004_edge2shoe_64/tests/test_batch_200i/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)


# ##### test batch for experience 003_edge2bag_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/003_edge2bag_64/outputs/edges2handbags_folder/checkpoints/gen_00800000.pt', type=str, help="checkpoint of autoencoders")

# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/003_edge2bag_64/tests/test_batch_200i/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/003_edge2bag_64/tests/test_batch_200i/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)


# ##### test batch for experience 016_MUNIT_origin_dog2catDRIT_64_cyc
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/016_MUNIT_origin_dog2catDRIT_64_cyc/outputs/dogs2catsDRIT_folder-cyc/checkpoints/gen_00300000.pt', type=str, help="checkpoint of autoencoders")

# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/Dog2CatDRIT/cat2dog/testA', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/016_MUNIT_origin_dog2catDRIT_64_cyc/tests/test_batch/a2b/a2b/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/Dog2CatDRIT/cat2dog/testB', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/016_MUNIT_origin_dog2catDRIT_64_cyc/tests/test_batch/b2a/b2a/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# ##### test batch for experience 017_MUNIT_origin_dog2cat_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/017_MUNIT_origin_dog2cat_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00300000.pt', type=str, help="checkpoint of autoencoders")

# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/Dog2CatDRIT/cat2dog/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/017_MUNIT_origin_dog2cat_64/tests/test_batch/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/Dog2CatDRIT/cat2dog/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/017_MUNIT_origin_dog2cat_64/tests/test_batch/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# ##### test batch for experience 012_MUNIT_origin_cityscapes_64_cyc
# # parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/gen_00800000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/gen_01000000.pt', type=str, help="checkpoint of autoencoders")

# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testA', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/a2b/a2b/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# # parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testB', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/from_MegaDeep/004_edge2shoe_64/tests/test_batch/b2a/b2a/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)


# ##### test batch for experience 015_cityscapes_64_cyc
# # parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/gen_00600000.pt', type=str, help="checkpoint of autoencoders")
# # parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/gen_00800000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/gen_01000000.pt', type=str, help="checkpoint of autoencoders")
#
# # parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/tests/test_batch/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/tests/test_batch/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# ##### test batch for experience 018_MUNIT_origin_dog2catDRIT_64
# # parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00300000.pt', type=str, help="checkpoint of autoencoders")
# # parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00400000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00500000.pt', type=str, help="checkpoint of autoencoders")

# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/Dog2CatDRIT/cat2dog/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/tests/test_batch_500k/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/Dog2CatDRIT/cat2dog/testB', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/tests/test_batch_500k/b2a/b2a/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# ##### test batch for experience 019_InfoMUNIT_dog2catDRIT_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00300000.pt', type=str, help="checkpoint of autoencoders")
# # parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00400000.pt', type=str, help="checkpoint of autoencoders")
# # parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00500000.pt', type=str, help="checkpoint of autoencoders")

# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/Dog2CatDRIT/cat2dog/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/tests/test_batch_500k/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/Dog2CatDRIT/cat2dog/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/tests/test_batch_300k/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# ##### test batch for experience 020_MUNIT_portrait_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/020_MUNIT_portrait_64/outputs/portrait_folder/checkpoints/gen_00700000.pt', type=str, help="checkpoint of autoencoders")

# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/portrait/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/020_MUNIT_portrait_64/tests/test_batch_700k/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/portrait/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/020_MUNIT_portrait_64/tests/test_batch_700k/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# ##### test batch for experience 022_InfoMUNIT_portrait_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/022_InfoMUNIT_portrait_64/outputs/portrait_folder/checkpoints/gen_00300000.pt', type=str, help="checkpoint of autoencoders")
# # parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/022_InfoMUNIT_portrait_64/outputs/portrait_folder/checkpoints/gen_00500000.pt', type=str, help="checkpoint of autoencoders")
# # parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/022_InfoMUNIT_portrait_64/outputs/portrait_folder/checkpoints/gen_00700000.pt', type=str, help="checkpoint of autoencoders")


# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/portrait/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/022_InfoMUNIT_portrait_64/tests/test_batch_300k/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/portrait/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/022_InfoMUNIT_portrait_64/tests/test_batch_300k/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

############### experiments after ECCV, before WACV
### Baseline: 003_edge2bag_64 checkpoint 800k

# # parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/024_InfoMUNIT_infoLen_4/outputs/edges2handbags_folder/checkpoints/gen_00500000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/024_InfoMUNIT_infoLen_4/outputs/edges2handbags_folder/checkpoints/gen_00800000.pt', type=str, help="checkpoint of autoencoders")
# #
# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/024_InfoMUNIT_infoLen_4/tests/test_batch_800k/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/024_InfoMUNIT_infoLen_4/tests/test_batch_800k/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/026_InfoMUNIT_infoLen_8/outputs/edges2handbags_folder/checkpoints/gen_00610000.pt', type=str, help="checkpoint of autoencoders")
# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/026_InfoMUNIT_infoLen_8/tests/test_batch_800k/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/026_InfoMUNIT_infoLen_8/tests/test_batch_800k/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/031_InfoMUNIT_losses_wo_x/outputs/edges2handbags_folder/checkpoints/gen_00660000.pt', type=str, help="checkpoint of autoencoders")
# # parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testA', type=str, help="input image folder")
# # parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/031_InfoMUNIT_losses_wo_x/tests/test_batch_660k/a2b/a2b/', type=str, help="output image folder")
# # parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Data/edges2handbags/testB', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/031_InfoMUNIT_losses_wo_x/tests/test_batch_660k/b2a/b2a/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)

parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/025_InfoMUNIT_infoLen_6/outputs/edges2handbags_folder/checkpoints/gen_00580000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/edges2handbags/testA', type=str, help="input image folder")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/025_InfoMUNIT_infoLen_6/tests/test_batch_580k/a2b/a2b/', type=str, help="output image folder")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/edges2handbags/testB', type=str, help="input image folder")
parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/025_InfoMUNIT_infoLen_6/tests/test_batch_580k/b2a/b2a/', type=str, help="output image folder")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=0)


opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Load the inception networks if we need to compute IS or CIIS
if opts.compute_IS or opts.compute_CIS:
    inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
    inception.cuda()
    # freeze the inception models and set eval mode
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder_test_batch(opts.input_folder, 1, False, new_size=config['new_size'], height=config['new_size'], width=config['new_size'], crop=True)

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
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
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if opts.compute_IS:
    IS = []
    all_preds = []
if opts.compute_CIS:
    CIS = []

if opts.trainer == 'MUNIT':
    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if i >= LIMIT_INPUT:
            break
        if opts.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)
        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            if opts.compute_IS or opts.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opts.compute_IS:
                all_preds.append(pred)
            if opts.compute_CIS:
                cur_preds.append(pred)
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder+"_%02d"%j,basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if opts.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    output_text = ""
    if opts.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
        output_text += ("Inception Score: {}".format(np.exp(np.mean(IS)))+"\n")
    if opts.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))
        output_text += ("conditional Inception Score: {}".format(np.exp(np.mean(CIS)))+"\n")

    if opts.compute_IS or opts.compute_CIS:
        with open(os.path.join(opts.output_folder, "results.txt"), "w+") as fp:
            fp.write(output_text)

# elif opts.trainer == 'UNIT':
#     # Start testing
#     for i, (images, names) in enumerate(zip(data_loader, image_names)):
#         print(names[1])
#         images = Variable(images.cuda(), volatile=True)
#         content, _ = encode(images)
#
#         outputs = decode(content)
#         outputs = (outputs + 1) / 2.
#         # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
#         basename = os.path.basename(names[1])
#         path = os.path.join(opts.output_folder,basename)
#         if not os.path.exists(os.path.dirname(path)):
#             os.makedirs(os.path.dirname(path))
#         vutils.save_image(outputs.data, path, padding=0, normalize=True)
#         if not opts.output_only:
#             # also save input images
#             vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
else:
    pass
