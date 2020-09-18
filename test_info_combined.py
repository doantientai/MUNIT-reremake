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

# import imageio

parser = argparse.ArgumentParser()

############################### Run one test for all exp for the paper AIM ECCV 2020
parser.add_argument('--config', default='configs/edges2handbags_folder.yaml', type=str)
path_InfoMUNIT_workshop = "/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/"

# # # ### 003_edge2bag_64
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/003_edge2bag_64/outputs/edges2handbags_folder/gen_00800000.pt', type=str)
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/edges2handbags/testA/194_AB.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/003_edge2bag_64/tests/test_latent_info/a2b/194', type=str)

# # ### 004_edge2shoe_64: 177, 100, 143, 10
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/004_edge2shoe_64/outputs/edges2shoes_folder/checkpoints/gen_00800000.pt', type=str)
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/edges2shoes/testA/10_AB.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/004_edge2shoe_64/tests/test_latent_info/a2b/10', type=str)

# # ### 005_MUNIT_origin_edge2bag_64 :100, 194, 23
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/005_MUNIT_origin_edge2bag_64/outputs/edges2handbags_folder/gen_00800000.pt', type=str)
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/edges2handbags/testA/100_AB.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/005_MUNIT_origin_edge2bag_64/tests/test_latent_info/a2b/100', type=str)

# # ### 006_MUNIT_origin_edge2shoe_64: 177, 100, 143, 10
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/006_MUNIT_origin_edge2shoe_64/outputs/edges2shoes_folder/checkpoints/gen_00800000.pt', type=str)
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/edges2shoes/testA/177_AB.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/006_MUNIT_origin_edge2shoe_64/tests/test_latent_info/a2b/177', type=str)

# # ### 018_MUNIT_origin_dog2catDRIT_64 499, 28 / 1382, 120
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00100000.pt', type=str)
# # parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# # parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/Dog2CatDRIT/cat2dog/testA/28.jpg')
# # parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/018_MUNIT_origin_dog2catDRIT_64/tests/test_latent_info/a2b/28', type=str)
# parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/Dog2CatDRIT/cat2dog/testB/120.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/018_MUNIT_origin_dog2catDRIT_64/tests/test_latent_info/b2a/120', type=str)

# # ### 019_InfoMUNIT_dog2catDRIT_64 499, 28 / 1382, 120
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/outputs/dogs2catsDRIT_folder/checkpoints/gen_00100000.pt', type=str)
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/Dog2CatDRIT/cat2dog/testA/499.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/019_InfoMUNIT_dog2catDRIT_64/tests/test_latent_info/a2b/499', type=str)
# # parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
# # parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/Dog2CatDRIT/cat2dog/testB/1382.jpg')
# # parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/019_InfoMUNIT_dog2catDRIT_64/tests/test_latent_info/b2a/1382', type=str)

# # ### 020_MUNIT_portrait_64
# # george-stubbs_portrait-of-richard-wedgewood
# # jean-francois-millet_louise-antoinette-feuardent-1841
# # 089330
# # 105059
# parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/020_MUNIT_portrait_64/outputs/portrait_folder/checkpoints/gen_00200000.pt', type=str)
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/portrait/testA/george-stubbs_portrait-of-richard-wedgewood.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/020_MUNIT_portrait_64/tests/test_latent_info/a2b/george-stubbs_portrait-of-richard-wedgewood', type=str)
# # parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
# # parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/portrait/testB/105059.jpg')
# # parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/020_MUNIT_portrait_64/tests/test_latent_info/b2a/105059', type=str)

# ### 022_InfoMUNIT_portrait_64
# george-stubbs_portrait-of-richard-wedgewood
# jean-francois-millet_louise-antoinette-feuardent-1841
# 089330
# 105059
parser.add_argument('--checkpoint', default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/022_InfoMUNIT_portrait_64/outputs/portrait_folder/checkpoints/gen_00200000.pt', type=str)
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/portrait/testA/jean-francois-millet_louise-antoinette-feuardent-1841.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/022_InfoMUNIT_portrait_64/tests/test_latent_info/a2b/jean-francois-millet_louise-antoinette-feuardent-1841', type=str)
parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/portrait/testB/089330.jpg')
parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/022_InfoMUNIT_portrait_64/tests/test_latent_info/b2a/089330', type=str)


# ## debug on local
# path_InfoMUNIT_workshop = "/media/tai/6TB/Projects/InfoMUNIT/"
# # parser.add_argument('--checkpoint', type=str,
# #                     default=path_InfoMUNIT_workshop + 'Models/ver_workshop/005_MUNIT_origin_edge2bag_64/outputs/edges2handbags_folder/gen_00800000.pt')
# # parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# # parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/edges2handbags/testA/23_AB.jpg')
# # parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/005_MUNIT_origin_edge2bag_64/tests/test_latent_info/a2b/23', type=str)

# parser.add_argument('--checkpoint', type=str, default=path_InfoMUNIT_workshop+'Models/003_edge2bag_64/outputs/edges2handbags_folder/checkpoints/gen_00800000.pt')
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default=path_InfoMUNIT_workshop+'Data/edges2handbags/testA/23_AB.jpg')
# parser.add_argument('--output_folder', default=path_InfoMUNIT_workshop+'Models/003_edge2bag_64/tests/test_latent_info/a2b/23', type=str)


parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style', type=int, default=20, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

# quali = {'quality': 100}
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

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def get_concat_h_loop(imgs):
    w = imgs[0].width
    h = imgs[0].height
    dst = Image.new('RGB', (w * len(imgs), h))
    for idx, img in enumerate(imgs):
        dst.paste(img, (w * idx, 0))
    return dst


def get_concat_v_loop(imgs):
    w = imgs[0].width
    h = imgs[0].height
    dst = Image.new('RGB', (w, h * len(imgs)))
    for idx, img in enumerate(imgs):
        dst.paste(img, (0, h * idx,))
    return dst


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
        ### test certain style code
        style_code_seed = Variable(torch.randn(1, style_dim, 1, 1).cuda())
        #         print(style_code_seed[0,:,0,0].cpu().numpy())
        #         exit()
        style_code_seed = torch.tensor(
            [[[[-0.7199]],
              [[-0.9169]],
              [[1.9330]],
              [[-0.8350]],
              [[-0.8820]],
              [[-0.7018]],
              [[0.0]],
              [[0.0]]]]).cuda()

        style_code = style_code_seed.clone()
        #         print(style_code)
        #         exit()
        imgs_digits = []
        # for code_position in range(style_dim):
            # code_values = [float(x)/5.0 - 2 for x in range(0, 21, 2)]
            #             code_values = [float(x)/20 - 1 for x in range(0, 41, 5)]  # 40 values, from -1 to +1
            #             code_values = [float(x)/20 - 1.5 for x in range(0, 61, 5)]  # 60 values, from -1.5 to +1.5
            #             code_values = [float(x)/20 - 2.0 for x in range(0, 81, 5)]  # 80 values, from -1.5 to +1.5
        code_values = [float(x) / 10 - 2.0 for x in range(0, 41, 5)]  # 40 values, from -1 to +1

        for k, code_value_k in enumerate(code_values):
            output_pillows = []
            for j, code_value_j in enumerate(code_values):
                style_code = style_code_seed.clone()
                style_code[0, 6, 0, 0] = code_value_j
                style_code[0, 7, 0, 0] = code_value_k
                # print(style_code[0, :, 0, 0])
                outputs = decode(content, style_code)
                outputs = (outputs + 1) / 2.
                path = os.path.join(opts.output_folder,
                                    file_name + '_out_info_%.1f_%.1f.jpg' % (code_value_j, code_value_k))
                # vutils.save_image(outputs.data, path, padding=0, normalize=True)
                output_pil = transforms.ToPILImage()(outputs[0].cpu())
                output_pillows.append(output_pil)
            # output_pillows[0].save(os.path.join(opts.output_folder, file_name + '_out_info.gif'),
            #                        save_all=True, append_images=output_pillows[1:], optimize=False, duration=80,
            #                        loop=0)
            imgs_digits.append(get_concat_h_loop(output_pillows))
        final_frame = get_concat_v_loop(imgs_digits)
        final_frame.save(os.path.join(opts.output_folder, file_name + '_info_combined.jpg'), quality=100)




    else:
        print('Only accept MUNIT trainer')

    # if not opts.output_only:
    # also save input images
    vutils.save_image(image.data, os.path.join(opts.output_folder, file_name + '_input.jpg'), padding=0, normalize=True)
