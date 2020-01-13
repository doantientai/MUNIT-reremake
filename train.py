"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
from random import random
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mnist2svhn_002_infoStyle.yaml', help='Path to the config file.')
# parser.add_argument('--output_path', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/MUNIT-reremake/Models/debug', help="output path server")
# parser.add_argument('--output_path', type=str, default='/home/tai/Desktop/MUNIT-reremake-log/debug', help="outputs path")
parser.add_argument('--output_path', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC_4l_LL10k_debug', help="outputs path")
parser.add_argument("--resume", action="store_true")
# parser.add_argument("--resume", default=True)
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True


def get_display_images(loader):
    # randomly take one image from each class
    list_images = []
    list_classes_to_take = [x for x in range(len(loader.dataset.classes))]
    while len(list_classes_to_take) > 0:
        i = int(random() * len(loader.dataset.targets))
        label = loader.dataset[i][1]
        if label in list_classes_to_take:
            image = loader.dataset[i][0]
            list_images.append(image)
            list_classes_to_take.remove(label)
            # print(list_classes_to_take)
    return torch.stack(list_images).cuda()
    # train_display_images_a = torch.stack([loader.dataset[i][0]]).cuda()


# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
# elif opts.trainer == 'UNIT':
#     trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()
train_loader_a, train_loader_a_limited, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
# train_display_images_a_temp = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
# train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
# test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
# test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in range(display_size)]).cuda()

train_display_images_a = get_display_images(train_loader_a)
train_display_images_b = get_display_images(train_loader_b)
test_display_images_a = get_display_images(test_loader_a)
test_display_images_b = get_display_images(test_loader_b)

# print(train_display_images_a.size())
# print([train_loader_a.dataset[i][1] for i in range(display_size)])
# exit()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (samples_a, samples_b, samples_a_limited) in enumerate(
            zip(train_loader_a, train_loader_b, train_loader_a_limited)):
        # images_a, labels_a = samples_a
        images_a, _ = samples_a
        # images_a_limited, labels_a_limited = samples_a_limited
        images_a_limited = samples_a_limited[0]
        images_b, labels_b = samples_b

        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        # labels_a, labels_b = labels_a.cuda().detach(), labels_b.cuda().detach()
        labels_b = labels_b.cuda().detach()
        # images_a_limited, labels_a_limited = images_a_limited.cuda().detach(), labels_a_limited.cuda().detach()
        images_a_limited = images_a_limited.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            # time_start_iter = time()

            trainer.dis_update(images_a, images_b, config)
            # time_dis = time()
            # print(f'Dis: {time_dis - time_start_iter}', end=" ")

            trainer.gen_update(images_a, [images_b, labels_b], config, [images_a_limited, 0])
            # time_gen = time()
            # print(f'Gen: {time_gen - time_dis}', end=" ")

            trainer.cla_update([images_a_limited, 0], [images_b, labels_b])
            # time_con_cla = time()
            # print(f'Cla: {time_con_cla - time_gen}')

            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            # write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
            trainer.cla_inference(test_loader_a, test_loader_b)

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
