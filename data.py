"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.utils.data as data
from torchvision.datasets.folder import DatasetFolder
import os.path
from random import shuffle
import h5py
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


# class ImageFilelist(data.Dataset):
#     def __init__(self, root, flist, transform=None,
#                  flist_reader=default_flist_reader, loader=default_loader):
#         self.root = root
#         self.imlist = flist_reader(flist)
#         self.transform = transform
#         self.loader = loader
#
#     def __getitem__(self, index):
#         impath = self.imlist[index]
#         img = self.loader(os.path.join(self.root, impath))
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img
#
#     def __len__(self):
#         return len(self.imlist)


# class ImageLabelFilelist(data.Dataset):
#     def __init__(self, root, flist, transform=None,
#                  flist_reader=default_flist_reader, loader=default_loader):
#         self.root = root
#         self.imlist = flist_reader(os.path.join(self.root, flist))
#         self.transform = transform
#         self.loader = loader
#         self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
#         self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
#         self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]
#
#     def __getitem__(self, index):
#         impath, label = self.imgs[index]
#         img = self.loader(os.path.join(self.root, impath))
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)


###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
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


class ImageFolderTorchVision(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, num_samples=None):
        super(ImageFolderTorchVision, self).__init__(root, loader, IMG_EXTENSIONS,
                                                     transform=transform,
                                                     target_transform=target_transform)
        if num_samples is None:
            self.targets = [x[1] for x in self.samples]
            self.imgs = self.samples
        else:
            shuffle(self.samples)
            self.samples = self.samples[:num_samples]
            self.targets = [x[1] for x in self.samples[:num_samples]]
            self.imgs = self.samples[:num_samples]


class H5Dataset(data.Dataset):
    def __init__(self, file_path, load_labels=False, transform=None):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.transform = transform
        self.images = h5_file.get('images')
        self.targets = None
        self.classes = []
        if load_labels:
            h5_file_labels = h5py.File(file_path.replace('images.h5', 'labels.h5'))
            self.targets = h5_file_labels.get('labels')
            self.classes = np.unique(self.targets)

    def __getitem__(self, index):
        img = self.images[index, :, :, :]
        label = self.targets[index]
        # label = torch.from_numpy(self.targets[index]).float()
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.images.shape[0]
