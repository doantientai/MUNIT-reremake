"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# from __future__ import print_function
# from utils import get_config, get_data_loader_folder_for_test, pytorch03_to_pytorch04, load_inception
from utils import get_config, get_data_loader_folder_for_test, pytorch03_to_pytorch04, get_all_data_loaders
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
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

parser.add_argument('--config', default='configs/mnist2svhn_002_infoStyle.yaml', type=str, help="net configuration")
parser.add_argument('--checkpoint_gen', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6lU_1shot/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
parser.add_argument('--checkpoint_con_cla', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC6lU_1shot/con_cla_00370000.pt', type=str, help="checkpoint of autoencoders")
parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/', type=str)


opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
# input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Setup model and data loader
# image_folder = ImageFolder(opts.input_folder, transform=None, return_paths=True)
# train_loader = get_data_loader_for_testing_encoder(config)

# exit()
# config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT")

# try:
state_dict_gen = torch.load(opts.checkpoint_gen)
trainer.gen_a.load_state_dict(state_dict_gen['a'])

state_dict_con_cla = torch.load(opts.checkpoint_con_cla)
trainer.content_classifier.load_state_dict(state_dict_con_cla['con_cla'])

# trainer.gen_b.load_state_dict(state_dict_gen['b'])
# except:
#     state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
#     trainer.gen_a.load_state_dict(state_dict['a'])
#     trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
config['batch_size_val'] = 64
config['batch_size'] = 64
train_loader_a, train_loader_a_limited, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

accu_content_classifier_c_a = []
# accu_content_classifier_c_b = []

for it_inf, samples_a_test in enumerate(test_loader_a):

    x_a, label_a = samples_a_test[0].cuda().detach(), samples_a_test[1].cuda().detach()
    # x_b, label_b = samples_b_test[0].cuda().detach(), samples_b_test[1].cuda().detach()

    # encode
    c_a, s_a_prime = trainer.gen_a.encode(x_a)
    # c_b, s_b_prime = trainer.gen_b.encode(x_b)

    label_predict_c_a = trainer.content_classifier(c_a)
    # label_predict_c_b = trainer.content_classifier(c_b)

    batch_accuracy = trainer.compute_content_classifier_accuracy(label_predict_c_a, label_a, custom_batch_size_val=config['batch_size_val']) * 100.0
    accu_content_classifier_c_a.append(batch_accuracy)

    print(it_inf, batch_accuracy)
    # accu_content_classifier_c_b.append(trainer.compute_content_classifier_accuracy(label_predict_c_b, label_b))

accu_content_classifier_c_a_mean = trainer.mean_list(accu_content_classifier_c_a)
# accu_content_classifier_c_b_mean = trainer.mean_list(accu_content_classifier_c_b)

print('Accuracy on test A: %.2f' % accu_content_classifier_c_a_mean)
# print('Accuracy on test B: %.2f', accu_content_classifier_c_b_mean)

# encode_a2b = trainer.gen_a.encode
# encode_b2a = trainer.gen_b.encode

# decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode  # decode function

# array_output = None
# LIMIT_SAMPLE = None
#
# hyperparameters = config
# content_domain_classifier = ContentClassifier(hyperparameters['gen']['dim'], hyperparameters)
# content_domain_classifier.cuda()
# content_classifier_params = list(content_domain_classifier.parameters())
#
# beta1 = hyperparameters['beta1']
# beta2 = hyperparameters['beta2']
# cla_opt = torch.optim.Adam([p for p in content_classifier_params if p.requires_grad], lr=hyperparameters['lr'],
#                            betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
#
# iterations = 0
# max_iter = 100000
# accu_log = 0
# if opts.trainer == 'MUNIT':
#     # Start testing
#     for iterations, batch in enumerate(train_loader):
#         # if LIMIT_SAMPLE is not None:
#         #     if i >= LIMIT_SAMPLE:
#         #         break
#
#         images, labels = batch
#         images = Variable(images.cuda(), volatile=True)
#
#         labels_np = labels.cpu().detach().numpy()
#         # contents, _ = encode_b2a(images)
#         if labels_np[0] == 1:
#             contents, _ = encode_b2a(images)
#         elif labels_np[0] == 0:
#             contents, _ = encode_a2b(images)
#         else:
#             contents = None
#             exit("Bad label:" + str(labels_np[0]))
#         images = Variable(images.cuda(), volatile=True)
#
#         contents_np = contents.cpu().detach().numpy()
#
#         ### train content domain classifier
#         cla_opt.zero_grad()
#         predicted_domain = content_domain_classifier(contents)
#         # loss = trainer.compute_content_classifier_loss(predicted_domain, torch.tensor(labels).cuda())
#         loss = trainer.compute_content_classifier_loss(predicted_domain, torch.tensor([iterations % 2]).cuda())
#         # accu_log += trainer.compute_content_classifier_accuracy(predicted_domain, torch.tensor(labels).cuda())
#         if np.argmax(predicted_domain.cpu().detach().numpy()) == labels_np[0]:
#             accu_log += 1
#
#         loss.backward()
#         cla_opt.step()
#
#         if (iterations + 1) % config['log_iter'] == 0:
#             accu_log_mean = accu_log / config['log_iter'] * 100
#             print("Loss: %.2f train_accuracy: %.2f Iteration: %08d/%08d" % (loss.cpu(), accu_log_mean, iterations + 1, max_iter))
#             accu_log_mean = None
#             accu_log = 0
#
#         iterations += 1
#         if iterations >= max_iter:
#             sys.exit('Finish training')
