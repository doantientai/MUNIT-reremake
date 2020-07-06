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
from data import is_image_file
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


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
parser.add_argument('--inception_a', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/inception_v3_google-1a9a5a14.pth', help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/inception_v3_google-1a9a5a14.pth', help="path to the pretrained inception network for domain B")
LIMIT_INPUT = 100

parser.add_argument('--checkpoint', default='', type=str, help="checkpoint of autoencoders")
parser.add_argument('--output_folder', default='', type=str, help="output image folder")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)

# ##### test batch for 001_portrait
# # parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/001_portrait/a2b', type=str, help="input image folder")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/001_portrait/b2a', type=str, help="input image folder")

##### test batch for 002_edges2handbags
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/002_edges2handbags/a2b', type=str, help="input image folder")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/002_edges2handbags/b2a', type=str, help="input image folder")

#### test batch for 003_edges2shoes
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/003_edges2shoes/a2b', type=str, help="input image folder")
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/003_edges2shoes/b2a', type=str, help="input image folder")

#### test batch for 004_cat2dogDRIT
# parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/004_cat2dogDRIT/a2b', type=str, help="input image folder")
parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/004_cat2dogDRIT/b2a', type=str, help="input image folder")

opts = parser.parse_args()

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and (fname != 'input.png'):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class ImageFolderNoInput(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def get_data_loader_folder_test_batch_DRIT(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.CenterCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolderNoInput(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


if __name__ == '__main__':
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    # Load experiment setting
    config = get_config(opts.config)
    input_dim = 3

    # Load the inception networks if we need to compute IS or CIIS
    if opts.compute_IS or opts.compute_CIS:
        inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
        inception.cuda()
        # freeze the inception models and set eval mode
        inception.eval()
        for param in inception.parameters():
            param.requires_grad = False
        inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

    list_samples = os.listdir(opts.input_folder)
    list_samples.sort()

    config['vgg_model_path'] = opts.output_path
    if opts.compute_IS:
        IS = []
        all_preds = []
    if opts.compute_CIS:
        CIS = []

    for output in list_samples:
        if opts.compute_CIS:
            cur_preds = []
        print(output)
        image_names = ImageFolderNoInput(os.path.join(opts.input_folder, output), transform=None, return_paths=True)
        data_loader = get_data_loader_folder_test_batch_DRIT(os.path.join(opts.input_folder, output), 1, False, new_size=config['new_size'],
                                                             height=config['new_size'], width=config['new_size'], crop=True)

        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            print(names[1])

            # this line does the trick (RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight')
            images = Variable(images.cuda(), volatile=True)

            outputs = images
            if opts.compute_IS or opts.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opts.compute_IS:
                all_preds.append(pred)
            if opts.compute_CIS:
                cur_preds.append(pred)
        if opts.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    output_text = ""
    if opts.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
        output_text += ("Inception Score: {}".format(np.exp(np.mean(IS))) + "\n")
    if opts.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))
        output_text += ("conditional Inception Score: {}".format(np.exp(np.mean(CIS))) + "\n")

    if opts.compute_IS or opts.compute_CIS:
        with open(os.path.join(opts.input_folder, "results.txt"), "w+") as fp:
            fp.write(output_text)
