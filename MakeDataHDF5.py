import h5py
import os
import cv2
import argparse
from utils import get_config
import numpy as np
from random import shuffle

# path_dir_in = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainA'
# path_dir_out = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT_h5/mnist2svhn_w_labels/trainA'
path_dir_in = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainA_50'
path_dir_out = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT_h5/mnist2svhn_w_labels/trainA_50'
# path_dir_in = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/trainB'
# path_dir_out = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT_h5/mnist2svhn_w_labels/trainB'
# path_dir_in = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/testB'
# path_dir_out = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT_h5/mnist2svhn_w_labels/testB'
# path_dir_in = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/testA'
# path_dir_out = '/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT_h5/mnist2svhn_w_labels/testA'

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mnist2svhn_002_infoStyle.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = get_config(opts.config)
new_size = config['new_size']

if __name__ == '__main__':
    os.makedirs(path_dir_out)

    list_paths = []
    list_images = []
    list_labels = []

    for (dir_path, dir_names, file_names) in os.walk(path_dir_in):
        for file in file_names:
            # get path
            im_path = os.path.join(dir_path, file)
            # print(im_path)
            list_paths.append(im_path)

    # with this small data, 2 loops is ok
    shuffle(list_paths)  # shuffle in place

    for im_path in list_paths:
        # get image
        image = cv2.imread(im_path)
        if image.shape != (new_size, new_size, 3):
            image = cv2.resize(image, (new_size, new_size))
        list_images.append(image)

        # get label
        label = im_path.split('/')[-2]
        list_labels.append(int(label))

    # print(list_paths[0])
    # print(list_images[0].shape)
    # print(list_labels[0])

    array_images = np.array(list_images)
    array_labels = np.array(list_labels)

    print("array_images:", array_images.shape)
    print('array_labels:', array_labels.shape)

    path_out_images = os.path.join(path_dir_out, 'images.h5')
    path_out_labels = os.path.join(path_dir_out, 'labels.h5')

    with h5py.File(path_out_images, 'w') as hf:
        hf.create_dataset("images", data=array_images)

    with h5py.File(path_out_labels, 'w') as hf:
        hf.create_dataset("labels", data=array_labels)
