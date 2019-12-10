"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, ContentClassifier
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import pytorch_msssim


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

        self.content_classifier = ContentClassifier(hyperparameters['gen']['dim'], hyperparameters)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        # dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())

        dis_named_params = list(self.dis_a.named_parameters()) + list(self.dis_b.named_parameters())
        # gen_named_params = list(self.gen_a.named_parameters()) + list(self.gen_b.named_parameters())

        ### modifying list params
        dis_params = list()
        # gen_params = list()
        for name, param in dis_named_params:
            if "_Q" in name:
                print('%s --> gen_params' % name)
                gen_params.append(param)
            else:
                dis_params.append(param)

        content_classifier_params = list(self.content_classifier.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.cla_opt = torch.optim.Adam([p for p in content_classifier_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.cla_scheduler = get_scheduler(self.cla_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        self.content_classifier.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.gan_type = hyperparameters['dis']['gan_type']
        self.criterionQ_con = NormalNLLLoss()

        self.criterion_content_classifier = nn.CrossEntropyLoss()

        self.batch_size = hyperparameters['batch_size']

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, sample_a, sample_b, hyperparameters):
        x_a, label_a = sample_a
        x_b, label_b = sample_b

        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        # self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        # self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        x_ba_dis_out = self.dis_a(x_ba)
        self.loss_gen_adv_a = self.compute_gen_adv_loss(x_ba_dis_out)

        # self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        x_ab_dis_out = self.dis_b(x_ab)
        self.loss_gen_adv_b = self.compute_gen_adv_loss(x_ab_dis_out)

        # loss info continuous
        self.info_cont_loss_a = self.compute_info_cont_loss(s_a, x_ba_dis_out)
        self.info_cont_loss_b = self.compute_info_cont_loss(s_b, x_ab_dis_out)

        # # domain-invariant perceptual loss
        # self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        # self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # ssim loss
        # self.loss_ssim_a = self.compute_ssim_loss(x_a, x_ab)
        # self.loss_ssim_b = self.compute_ssim_loss(x_b, x_ba)

        # total loss

        label_predict_c_a = self.content_classifier(c_a)
        label_predict_c_a_recon = self.content_classifier(c_a_recon)
        label_predict_c_b = self.content_classifier(c_b)
        label_predict_c_b_recon = self.content_classifier(c_b_recon)

        loss_content_classifier_c_a = self.compute_content_classifier_loss(label_predict_c_a, label_a)
        loss_content_classifier_c_a_recon = self.compute_content_classifier_loss(label_predict_c_a_recon, label_a)

        loss_content_classifier_c_a_and_c_a_recon = self.compute_content_classifier_two_predictions_loss(label_predict_c_a_recon, label_predict_c_a)
        loss_content_classifier_c_b_and_c_b_recon = self.compute_content_classifier_two_predictions_loss(label_predict_c_b_recon, label_predict_c_b)

        # loss_content_classifier_b = self.compute_content_classifier_loss(label_predict_c_b, label_b)
        # loss_content_classifier_c_b_recon = self.compute_content_classifier_loss(label_predict_c_b_recon, label_b)

        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              self.info_cont_loss_a + \
                              self.info_cont_loss_b + \
                              loss_content_classifier_c_a + \
                              loss_content_classifier_c_a_recon + \
                              loss_content_classifier_c_a_and_c_a_recon + \
                              loss_content_classifier_c_b_and_c_b_recon
                              # loss_content_classifier_b + \
                              # loss_content_classifier_c_b_recon
        # hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
        # hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
        # hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
        # hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
        # 0.1 * self.loss_ssim_a + \
        # 0.1 * self.loss_ssim_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    # def compute_ssim_loss(self, tensor_1, tensor_2):
    #     # print("tensor_1")
    #     # print(tensor_1.size())
    #     # print("tensor_2")
    #     # print(tensor_2.size())
    #
    #     tensor_1_gray = tensor_1.mean(1)
    #     tensor_1_gray = tensor_1_gray.unsqueeze(1)
    #
    #     tensor_2_gray = tensor_2.mean(1)
    #     tensor_2_gray = tensor_2_gray.unsqueeze(1)
    #
    #     # print("tensor_1_gray")
    #     # print(tensor_1_gray.size())
    #     # print("tensor_2_gray")
    #     # print(tensor_2_gray.size())
    #
    #     # exit()
    #
    #     # loss = 0
    #     # for image_1, image_2 in zip(tensor_1, tensor_2):
    #     #     loss += pytorch_msssim.msssim(image_1.unsqueeze(0), image_2.unsqueeze(0), normalize=True)
    #     # print('loss:', loss.cpu().detach().numpy())
    #     ### on shot (let's see what will happen)
    #
    #     loss = pytorch_msssim.msssim(tensor_1_gray, tensor_2_gray, normalize=True)
    #     return loss

    def compute_info_cont_loss(self, style_code, outs_fake):
        loss = 0
        num_cont_code = 2
        for it, (out_fake) in enumerate(outs_fake):
            q_mu = out_fake['mu']
            q_var = out_fake['var']
            info_noise = style_code[:, -num_cont_code:].view(-1, num_cont_code).squeeze().squeeze()
            loss += self.criterionQ_con(info_noise, q_mu, q_var) * 0.1
        return loss

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def cla_update(self, sample_a, sample_b, hyperparameters):
        x_a, label_a = sample_a
        x_b, label_b = sample_b
        # print('cla_update')
        # print(x_a.device())
        # exit()
        self.cla_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # print("c_a")
        # print(c_a.size())
        # exit()
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        label_predict_c_a = self.content_classifier(c_a)
        label_predict_c_a_recon = self.content_classifier(c_a_recon)
        label_predict_c_b = self.content_classifier(c_b)
        label_predict_c_b_recon = self.content_classifier(c_b_recon)

        self.loss_content_classifier_c_a = self.compute_content_classifier_loss(label_predict_c_a, label_a)
        self.loss_content_classifier_c_a_recon = self.compute_content_classifier_loss(label_predict_c_a_recon, label_a)
        self.loss_content_classifier_c_a_and_c_a_recon = self.compute_content_classifier_two_predictions_loss(label_predict_c_a_recon, label_predict_c_a)

        # self.loss_content_classifier_b = self.compute_content_classifier_loss(label_predict_c_b, label_b)
        # self.loss_content_classifier_c_b_recon = self.compute_content_classifier_loss(label_predict_c_b_recon, label_b)
        self.loss_content_classifier_c_b_and_c_b_recon = self.compute_content_classifier_two_predictions_loss(label_predict_c_b_recon,
                                                                                         label_predict_c_b)

        self.accu_content_classifier_c_a = self.compute_content_classifier_accuracy(label_predict_c_a, label_a)
        self.accu_content_classifier_c_a_recon = self.compute_content_classifier_accuracy(label_predict_c_a_recon,
                                                                                          label_a)
        self.accu_content_classifier_c_b = self.compute_content_classifier_accuracy(label_predict_c_b, label_b)
        self.accu_content_classifier_c_b_recon = self.compute_content_classifier_accuracy(label_predict_c_b_recon,
                                                                                          label_b)
        self.accu_CC_all = self.mean_list([
            self.accu_content_classifier_c_a,
            self.accu_content_classifier_c_a_recon,
            self.accu_content_classifier_c_b,
            self.accu_content_classifier_c_b_recon
        ])

        self.loss_cla_total = self.loss_content_classifier_c_a + self.loss_content_classifier_c_a_recon + \
                              self.loss_content_classifier_c_a_and_c_a_recon + \
                              self.loss_content_classifier_c_b_and_c_b_recon
                              # self.loss_content_classifier_b + self.loss_content_classifier_c_b_recon + \
        self.loss_cla_total.backward()
        self.cla_opt.step()

    @staticmethod
    def mean_list(lst):
        return sum(lst) / len(lst)

    def dis_update(self, x_a, x_b, hyperparameters):
        # print('dis_update')
        # print(x_a.is_cuda())
        # exit()
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        # D loss
        # self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        x_ba_dis_out = self.dis_a(x_ba.detach())
        x_a_dis_out = self.dis_a(x_a)
        self.loss_dis_a = self.compute_dis_loss(x_ba_dis_out, x_a_dis_out)
        # self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        x_ab_dis_out = self.dis_b(x_ab.detach())
        x_b_dis_out = self.dis_b(x_b)
        self.loss_dis_b = self.compute_dis_loss(x_ab_dis_out, x_b_dis_out)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def compute_content_classifier_loss(self, label_predict, label_true):
        loss = self.criterion_content_classifier(label_predict, label_true)
        return loss

    def compute_content_classifier_two_predictions_loss(self, label_predict_1, label_predict_2):
        # loss = self.criterion_content_classifier(label_predict, label_true)
        # print(label_predict_1.size())
        # print(label_predict_2.size())
        loss = torch.mean(torch.abs(label_predict_1 - label_predict_2))
        # print(loss.size())
        # exit()
        return loss

    def compute_content_classifier_accuracy(self, label_predict, label_true):
        # print("label_true")
        # print(label_true)
        #
        # print("label_predict")
        # print(label_predict[0])
        # print("max")
        values, indices = label_predict.max(1)
        # print(indices)

        results = (label_true == indices)
        # print(results)

        total_correct = results.sum().cpu().numpy()
        # print("total_correct")
        # print(total_correct)

        # total_samples = results.size()
        # print("total_samples")
        # print(total_samples)

        accuracy = float(total_correct) / float(self.batch_size)
        # print("accuracy")
        # print(accuracy)
        #
        # exit()
        return accuracy

    def compute_dis_loss(self, outs_fake, outs_real):
        # calculate the loss to train D
        # outs0 = self.forward(input_fake)
        # outs1 = self.forward(input_real)
        loss = 0
        for it, (out_fake, out_real) in enumerate(zip(outs_fake, outs_real)):
            out_fake = out_fake['output_d']
            out_real = out_real['output_d']
            if self.gan_type == 'lsgan':
                loss += torch.mean((out_fake - 0) ** 2) + torch.mean((out_real - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out_fake.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out_real.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out_real), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def compute_gen_adv_loss(self, outs_fake):
        # calculate the loss to train G
        # out_fake = self.forward(input_fake)
        loss = 0
        for it, (out_fake) in enumerate(outs_fake):
            out_fake = out_fake['output_d']
            if self.gan_type == 'lsgan':
                loss += torch.mean((out_fake - 1) ** 2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out_fake.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
