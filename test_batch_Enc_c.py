"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# from __future__ import print_function
# from utils import get_config, get_data_loader_folder_for_test, pytorch03_to_pytorch04, load_inception
from utils import get_config, get_data_loader_folder_for_test, pytorch03_to_pytorch04, get_data_loader_for_testing_encoder
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
# parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
parser.add_argument('--config', default='configs/mnist2svhn_002_infoStyle.yaml', type=str, help="net configuration")
parser.add_argument('--output_only', default=True, help="whether only save the output images or also save the input images")

parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_CC4l_1shot/gen_00370000.pt', type=str, help="checkpoint of autoencoders")
parser.add_argument('--input_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn_w_labels/', type=str)
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Data/Decoded/', type=str, help="output image path")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
# input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Setup model and data loader
# image_folder = ImageFolder(opts.input_folder, transform=None, return_paths=True)
train_loader = get_data_loader_for_testing_encoder(config)

# exit()
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
encode_a2b = trainer.gen_a.encode
encode_b2a = trainer.gen_b.encode
# decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode  # decode function

array_output = None
LIMIT_SAMPLE = None


class ContentClassifier(nn.Module):
    # Classifier which is built for classifying on the content space
    def __init__(self, input_dim, params):
        super().__init__()
        self.n_layers = 2
        self.n_classes = 2
        self.dim = input_dim
        self.cnn = self._make_cnn()

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    def _make_cnn(self):
        dim = self.dim
        for i in range(self.n_layers - 1):
            dim *= 2
        cnn = []
        cnn += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn += [self.Flatten()]
        # cnn += [nn.Linear(1*16*16, self.n_classes)]
        # cnn += [nn.Linear(1024, self.n_classes)]  #
        cnn += [nn.Linear(256, self.n_classes)]  #
        cnn += [nn.Softmax()]
        cnn = nn.Sequential(*cnn)
        return cnn

    def forward(self, x):
        result = self.cnn(x)
        return result


hyperparameters = config
content_domain_classifier = ContentClassifier(hyperparameters['gen']['dim'], hyperparameters)
content_domain_classifier.cuda()
content_classifier_params = list(content_domain_classifier.parameters())

beta1 = hyperparameters['beta1']
beta2 = hyperparameters['beta2']
cla_opt = torch.optim.Adam([p for p in content_classifier_params if p.requires_grad], lr=hyperparameters['lr'],
                           betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

iterations = 0
max_iter = 100000
accu_log = 0
if opts.trainer == 'MUNIT':
    # Start testing
    for iterations, batch in enumerate(train_loader):
        # if LIMIT_SAMPLE is not None:
        #     if i >= LIMIT_SAMPLE:
        #         break

        images, labels = batch
        images = Variable(images.cuda(), volatile=True)

        labels_np = labels.cpu().detach().numpy()
        # contents, _ = encode_b2a(images)
        if labels_np[0] == 1:
            contents, _ = encode_b2a(images)
        elif labels_np[0] == 0:
            contents, _ = encode_a2b(images)
        else:
            contents = None
            exit("Bad label:" + str(labels_np[0]))
        images = Variable(images.cuda(), volatile=True)

        contents_np = contents.cpu().detach().numpy()

        ### train content domain classifier
        cla_opt.zero_grad()
        predicted_domain = content_domain_classifier(contents)
        # loss = trainer.compute_content_classifier_loss(predicted_domain, torch.tensor(labels).cuda())
        loss = trainer.compute_content_classifier_loss(predicted_domain, torch.tensor([iterations % 2]).cuda())
        # accu_log += trainer.compute_content_classifier_accuracy(predicted_domain, torch.tensor(labels).cuda())
        if np.argmax(predicted_domain.cpu().detach().numpy()) == labels_np[0]:
            accu_log += 1

        loss.backward()
        cla_opt.step()

        if (iterations + 1) % config['log_iter'] == 0:
            accu_log_mean = accu_log / config['log_iter'] * 100
            print("Loss: %.2f train_accuracy: %.2f Iteration: %08d/%08d" % (loss.cpu(), accu_log_mean, iterations + 1, max_iter))
            accu_log_mean = None
            accu_log = 0

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
