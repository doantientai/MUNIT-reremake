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
import imageio

parser = argparse.ArgumentParser()

# ### 003_edge2bag_64/gen_00800000
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/from_MegaDeep/003_edge2bag_64/gen_00800000.pt', type=str)
# parser.add_argument('--config', default='configs/edges2handbags_folder.yaml', type=str)
#
# # parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
# # parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/ver_workshop/Untitled Folder/testB/90_AB.jpg')
# # parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/from_MegaDeep/003_edge2bag_64/tests/test_latent/b2a/90_AB', type=str)
#
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/ver_workshop/Untitled Folder/testA/32_AB.jpg')
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/from_MegaDeep/003_edge2bag_64/tests/test_latent/a2b/32_AB', type=str)

# ### 004_edge2shoe_64/gen_00800000
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/from_MegaDeep/004_edge2shoe_64/gen_00800000.pt', type=str)
# parser.add_argument('--config', default='configs/edges2handbags_folder.yaml', type=str)
#
# # parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
# # parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Source/InfoMUNIT-workshop/MUNIT-reremake/datasets/edges2shoes/testB/95_AB.jpg')
# # parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/from_MegaDeep/004_edge2shoe_64/tests/test_latent/b2a/95_AB', type=str)
#
# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Source/InfoMUNIT-workshop/MUNIT-reremake/datasets/edges2shoes/testA/107_AB.jpg')
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/from_MegaDeep/004_edge2shoe_64/tests/test_latent/a2b/107_AB', type=str)

### 012_MUNIT_origin_cityscapes_64_cyc/gen_00800000
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/gen_00800000.pt', type=str)
parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/gen_01000000.pt', type=str)
parser.add_argument('--config', default='configs/edges2handbags_folder.yaml', type=str)

# parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testA/60.jpg')
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_latent/a2b/60', type=str)
parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testB/63.jpg')
parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_latent/b2a/63', type=str)

# # ### 015_cityscapes_64_cyc/gen_00600000
# # parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/gen_00600000.pt', type=str)
#
# ### 015_cityscapes_64_cyc/gen_00800000
# parser.add_argument('--checkpoint', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/gen_00800000.pt', type=str)
# parser.add_argument('--config', default='configs/edges2handbags_folder.yaml', type=str)
#
# # parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
# # parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testA/60.jpg')
# # parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/tests/test_latent/a2b/60', type=str)
# parser.add_argument('--a2b', type=int, default=0, help="1 for a2b and 0 for b2a")
# parser.add_argument('--input', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testB/63.jpg')
# parser.add_argument('--output_folder', default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_latent/b2a/63', type=str)


parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style', type=int, default=20, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

quali = {'quality': 90}
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
        ### test certain style code
        style_code_seed = Variable(torch.randn(1, style_dim, 1, 1).cuda())
        style_code = style_code_seed.clone()

        for code_position in range(style_dim):
            # code_values = [float(x)/5.0 - 2 for x in range(0, 21, 2)]
            code_values = [float(x)/20 - 1 for x in range(0, 41, 1)]  # 40 values, from -1 to +1

            output_pillows = []
            # for j, code_value in enumerate(code_values):
            #     style_code[0, code_position, 0, 0] = code_value
            #     outputs = decode(content, style_code)
            #     outputs = (outputs + 1) / 2.
            #     path = os.path.join(opts.output_folder,
            #                         file_name + '_out_code%d_j%s_val_%.1f.jpg' % (code_position, str(j).zfill(2), code_value))
            #     # vutils.save_image(outputs.data, path, padding=0, normalize=True)
            #     output_pil = transforms.ToPILImage()(outputs[0].cpu())
            #     # output_pil.save(path)
            #     # output_pillows.append(output_pil)
            #     output_pillows.append(outputs[0].cpu().numpy())
            #
            # # output_pillows[0].save(os.path.join(opts.output_folder,  file_name + '_out_code%d.gif' % code_position),
            # #                        save_all=True, append_images=output_pillows[1:], optimize=True, duration=80, loop=0)
            # # imageio.mimsave(os.path.join(opts.output_folder,  file_name + '_out_code%d.gif' % code_position),
            # #                 output_pillows)
            # gif_path = os.path.join(opts.output_folder,  file_name + '_out_code%d.gif' % code_position)
            # with imageio.get_writer(gif_path, mode='I') as writer:
            #     for frame in output_pillows:
            #         writer.append_data(frame)

            for j, code_value in enumerate(code_values):
                style_code[0, code_position, 0, 0] = code_value
                outputs = decode(content, style_code)
                outputs = (outputs + 1) / 2.
                path = os.path.join(opts.output_folder,
                                    file_name + '_out_code%d_j%s_val_%.1f.jpg' % (
                                    code_position, str(j).zfill(2), code_value))
                # vutils.save_image(outputs.data, path, padding=0, normalize=True)
                output_pil = transforms.ToPILImage()(outputs[0].cpu())
                # output_pil.save(path)
                output_pillows.append(output_pil)
            output_pillows[0].save(os.path.join(opts.output_folder, file_name + '_out_code%d.gif' % code_position),
                                   save_all=True, append_images=output_pillows[1:], optimize=False, duration=80,
                                   loop=0)



    else:
        print('Only accept MUNIT trainer')

    # if not opts.output_only:
        # also save input images
    vutils.save_image(image.data, os.path.join(opts.output_folder, file_name+'_input.jpg'), padding=0, normalize=True)
