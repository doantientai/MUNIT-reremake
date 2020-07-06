"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
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

# ##### test batch for experience 001_edge2shoe
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/001_edge2shoe/test_latest/fakeB', type=str, help="input image folder")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/001_edge2shoe/test_latest/fakeA', type=str, help="input image folder")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/004_portrait/test_latest/fakeA', type=str, help="input image folder")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/004_portrait/test_latest/fakeB', type=str, help="input image folder")
<<<<<<< HEAD
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/SOTAs/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/002_edge2bag/test_latest/fakeA', type=str, help="input image folder")
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/SOTAs/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/002_edge2bag/test_latest/fakeB', type=str, help="input image folder")
# parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/SOTAs/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/002_edge2bag/test_latest/fakeA', type=str, help="input image folder")

### test output of MUNIT using script test_batch_cycleGAN
parser.add_argument('--input_folder', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/tests/test_batch_300k/a2b/a2b/103', type=str, help="input image folder")



=======
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/003_cat2dog/test_latest/fakeA', type=str, help="input image folder")
parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/CycleGAN/pytorch-CycleGAN-and-pix2pix/tests/003_cat2dog/test_latest/fakeB', type=str, help="input image folder")
>>>>>>> 12c53beadf2b3897abfd3f20e98bde518baf2a28
opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = 64

# Load the inception networks if we need to compute IS or CIIS
if opts.compute_IS or opts.compute_CIS:
    inception = load_inception('/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/inception_v3_google-1a9a5a14.pth')
    inception.cuda()
    # freeze the inception models and set eval mode
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder, 1, False, new_size=config['new_size'], crop=False)

IS = []
all_preds = []

if opts.trainer == 'MUNIT':
    # Start testing
    # style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        outputs = images
        # outputs = (outputs + 1) / 2.
        pred = F.softmax(inception(inception_up(outputs)),
                         dim=1).cpu().data.numpy()  # get the predicted class distribution

        all_preds.append(pred)

    all_preds = np.concatenate(all_preds, 0)
    py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
    for j in range(all_preds.shape[0]):
        pyx = all_preds[j, :]
        IS.append(entropy(pyx, py))

    output_text = ""
    print("Inception Score: {}".format(np.exp(np.mean(IS))))
    output_text += ("Inception Score: {}".format(np.exp(np.mean(IS)))+"\n")

    with open(os.path.join(opts.input_folder + "_results.txt"), "w+") as fp:
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
