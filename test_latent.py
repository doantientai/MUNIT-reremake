"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
# from trainer import MUNIT_Trainer, UNIT_Trainer
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
# parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn/testA/0.jpg', help="input image path")
parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn/testB/0.jpg', help="input image path")
# parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn/testB/20031.jpg', help="input image path")
# parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn/testA/8952.jpg', help="input image path")

# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT/MNIST2SVHN/train023_infoStyle_QaD_con_c4/output_test', type=str, help="output image path")
# parser.add_argument('--config', default='configs/mnist2svhn_002_infoStyle.yaml', type=str, help="net configuration")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT/MNIST2SVHN/train040_infoStyle_QaD_con_c2/outputs/checkpoints/gen_01000000.pt', type=str, help="checkpoint of autoencoders")

# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT/MNIST2SVHN/train_w_conc/train041_infoStyle_QaD_con_c2x10/output_test', type=str, help="output image path")
# parser.add_argument('--config', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT/MNIST2SVHN/train_w_conc/train041_infoStyle_QaD_con_c2x10/outputs/config.yaml', type=str, help="net configuration")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT/MNIST2SVHN/train_w_conc/train041_infoStyle_QaD_con_c2x10/outputs/checkpoints/gen_01000000.pt', type=str, help="checkpoint of autoencoders")

parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_Q_inG_noflip/output_test_latent', type=str, help="output image path")
parser.add_argument('--config', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_Q_inG_noflip/config.yaml', type=str, help="net configuration")
parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT-reremake/MUNIT_Q_inG_noflip/gen_00550000.pt', type=str, help="checkpoint of autoencoders")

# parser.add_argument('--num_style', type=int, default=8, help="number of styles to sample")
# parser.add_argument('--num_con_c', type=int, default=2, help="number of styles to sample")
# parser.add_argument('--config', default='configs/mnist2svhn_labelkeep_001.yaml', type=str, help="net configuration")
# parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/ForMUNIT/mnist2svhn/testA/0.jpg', help="input image path")
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT/MNIST2SVHN/train023_infoStyle_QaD_con_c4/output_test', type=str, help="output image path")
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/MUNIT/MNIST2SVHN/train023_infoStyle_QaD_con_c4/outputs/checkpoints/gen_01000000.pt', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--num_style', type=int, default=10, help="number of styles to sample")
# parser.add_argument('--config', type=str, help="net configuration")
# parser.add_argument('--input', type=str, help="input image path")
# parser.add_argument('--output_folder', type=str, help="output image path")
# parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style', type=int, default=20, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
# parser.add_argument('--output_only', default=True, help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
# elif opts.trainer == 'UNIT':
#     trainer = UNIT_Trainer(config)
else:
    # sys.exit("Only support MUNIT|UNIT")
    sys.exit("Only support MUNIT")

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
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode  # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode  # decode function

file_name = opts.input.split('/')[-1]
file_name = file_name[:-4]
# print(file_name)
# exit()

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b == 1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Variable(transform(Image.open(opts.input).convert('RGB')).unsqueeze(0).cuda())
    style_image = Variable(
        transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

    # Start testing
    content, _ = encode(image)

    if opts.trainer == 'MUNIT':
        style_code = Variable(torch.randn(1, style_dim - config['dis']['num_con_c'], 1, 1).cuda())

        # con_c_1 = Variable(torch.randn(opts.num_style, 1, 1, 1).cuda())
        # con_c_2 = Variable(torch.randn(1, 1, 1, 1).cuda())

        # s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # if num_con_c != 0:
        # con_c_a = Variable(torch.rand(opts.num_style, opts.num_con_c, 1, 1) * 2 - 1).cuda()
        #
        # s_a_info = torch.cat((style_rand, con_c_a), dim=1)

        # if opts.style != '':
        #     _, style = style_encode(style_image)
        # else:
        #     style = style_rand

        # style = style_rand
        con_c_1 = 0.0
        con_c_2 = 0.0

        con_c_1 -= 2.4
        # con_c_2 -= 2
        for j in range(11):
            # s = style[j].unsqueeze(0)
            con_c_1 += 0.4
            # con_c_2 += 0.1

            s = torch.cat([style_code, Variable(torch.tensor(con_c_1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda())], dim=1)
            s = torch.cat([s, Variable(torch.tensor(con_c_2).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda())], dim=1)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, file_name+'_output%s_conc1_%.1f.jpg'%(str(j).zfill(2), con_c_1))
            # path = os.path.join(opts.output_folder, file_name+'_output%s_conc2_%.1f.jpg'%(str(j).zfill(2), con_c_2))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)

    else:
        print('Only accept MUNIT trainer')

    # if not opts.output_only:
        # also save input images
    vutils.save_image(image.data, os.path.join(opts.output_folder, file_name+'_input.jpg'), padding=0, normalize=True)
