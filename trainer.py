"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, ContentDigitClassifier, ContentDomainClassifier
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
# import pytorch_msssim


# class NormalNLLLoss:
#     """
#     Calculate the negative log likelihood
#     of normal distribution.
#     This needs to be minimised.
#
#     Treating Q(cj | x) as a factored Gaussian.
#     """
#
#     def __call__(self, x, mu, var):
#         A = -0.5 * (var.mul(2 * np.pi) + 1e-6).log()
#         B = (x - mu)
#         C = B.pow(2).div(var.mul(2.0) + 1e-6)
#         logli = A - C
#         nll = -(logli.sum(1).mean())
#
#         return nll


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

        self.content_digit_classifier = ContentDigitClassifier(hyperparameters['gen']['dim'], hyperparameters)
        self.content_domain_classifier = ContentDomainClassifier(hyperparameters['gen']['dim'], hyperparameters)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.num_con_c = hyperparameters['dis']['num_con_c']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        # dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        # split params of the auto-encoder
        enc_style_params = list()
        enc_content_params = list()
        dec_params = list()
        # gen_params = list()
        for name, param in (list(self.gen_a.named_parameters()) + list(self.gen_b.named_parameters())):
            if "enc_style.model" in name:
                enc_style_params.append(param)
            elif "enc_content.model" in name:
                enc_content_params.append(param)
            elif ("dec.model" in name) or ("mlp.model" in name):
                dec_params.append(param)
            else:
                raise("weird param", name)

        dis_named_params = list(self.dis_a.named_parameters()) + list(self.dis_b.named_parameters())
        # gen_named_params = list(self.gen_a.named_parameters()) + list(self.gen_b.named_parameters())

        ### modifying list params
        dis_params = list()
        for name, param in dis_named_params:
            if "_Q" in name:
                # print('%s --> gen_params' % name)
                dec_params.append(param)
                gen_params.append(param)
            else:
                dis_params.append(param)

        content_digit_classifier_params = list(self.content_digit_classifier.parameters())
        content_domain_classifier_params = list(self.content_domain_classifier.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.enc_style_opt = torch.optim.Adam([p for p in enc_style_params if p.requires_grad],
                                              lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.enc_content_opt = torch.optim.Adam([p for p in enc_content_params if p.requires_grad],
                                                lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dec_opt = torch.optim.Adam([p for p in dec_params if p.requires_grad], lr=lr, betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])
        self.cla_digit_opt = torch.optim.Adam([p for p in content_digit_classifier_params if p.requires_grad],
                                              lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.cla_domain_opt = torch.optim.Adam([p for p in content_domain_classifier_params if p.requires_grad],
                                               lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.enc_style_scheduler = get_scheduler(self.enc_style_opt, hyperparameters)
        self.enc_content_scheduler = get_scheduler(self.enc_content_opt, hyperparameters)
        self.dec_scheduler = get_scheduler(self.dec_opt, hyperparameters)
        self.cla_digit_scheduler = get_scheduler(self.cla_digit_opt, hyperparameters)
        self.cla_domain_scheduler = get_scheduler(self.cla_domain_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        self.content_digit_classifier.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.gan_type = hyperparameters['dis']['gan_type']
        # self.criterionQ_con = NormalNLLLoss()

        self.criterion_content_classifier = nn.CrossEntropyLoss()

        # self.batch_size = hyperparameters['batch_size']
        self.batch_size_val = hyperparameters['batch_size_val']

        # self.accu_content_classifier_c_a = 0
        # self.accu_content_classifier_c_a_recon = 0
        # self.accu_content_classifier_c_b = 0
        # self.accu_content_classifier_c_b_recon = 0
        # self.accu_CC_all = 0

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

    def gen_update(self, x_a, sample_b, hyperparameters, sample_a_limited):
        x_b, label_b = sample_b
        x_a_limited, label_a_limited = sample_a_limited

        self.gen_opt.zero_grad()

        # self.enc_style_opt.zero_grad()
        # self.enc_content_opt.zero_grad()

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
        # x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        # x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        # GAN loss
        # self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        x_ba_dis_out = self.dis_a(x_ba)
        self.loss_gen_adv_a = self.compute_gen_adv_loss(x_ba_dis_out)

        # self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        x_ab_dis_out = self.dis_b(x_ab)
        self.loss_gen_adv_b = self.compute_gen_adv_loss(x_ab_dis_out)

        # label_predict_c_a = self.content_classifier(c_a)
        # label_predict_c_a_recon = self.content_classifier(c_a_recon)
        c_a_limited, _ = self.gen_a.encode(x_a_limited)
        label_predict_c_a_limited = self.content_digit_classifier(c_a_limited)

        label_predict_c_b = self.content_digit_classifier(c_b)
        label_predict_c_b_recon = self.content_digit_classifier(c_b_recon)

        x_ab_limited = self.gen_b.decode(c_a_limited, s_b)
        c_a_recon_limited, _ = self.gen_b.encode(x_ab_limited)
        label_predict_c_a_recon_limited = self.content_digit_classifier(c_a_recon_limited)

        ### loss content prediction a
        loss_content_classifier_c_a = self.compute_content_classifier_loss(label_predict_c_a_limited, label_a_limited)
        loss_content_classifier_c_a_recon = self.compute_content_classifier_loss(label_predict_c_a_recon_limited,
                                                                                 label_a_limited)

        ### loss content prediction b
        loss_content_classifier_c_b = self.compute_content_classifier_loss(label_predict_c_b, label_b)
        loss_content_classifier_c_b_recon = self.compute_content_classifier_loss(label_predict_c_b_recon, label_b)

        ### consistency of content prediction
        label_predict_c_a_unlabeled = self.content_digit_classifier(c_a)
        label_predict_c_a_unlabeled_recon = self.content_digit_classifier(c_a_recon)
        loss_content_classifier_c_a_and_c_a_recon = self.compute_content_classifier_two_predictions_loss(
            label_predict_c_a_unlabeled, label_predict_c_a_unlabeled_recon)
        loss_content_classifier_c_b_and_c_b_recon = self.compute_content_classifier_two_predictions_loss(
            label_predict_c_b, label_predict_c_b_recon)

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              loss_content_classifier_c_a + \
                              loss_content_classifier_c_a_recon + \
                              loss_content_classifier_c_b + \
                              loss_content_classifier_c_b_recon +\
                              loss_content_classifier_c_a_and_c_a_recon + \
                              loss_content_classifier_c_b_and_c_b_recon
        self.loss_gen_total.backward()
        self.gen_opt.step()

        self.dec_opt.zero_grad()

    # def dec_update_for_info_loss(self, x_a, sample_b, hyperparameters, sample_a_limited):
    #     x_b, label_b = sample_b
    #     # x_a_limited, label_a_limited = sample_a_limited
    #     self.dec_opt.zero_grad()
    #
    #     s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
    #     s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
    #
    #     c_b, s_b_prime = self.gen_b.encode(x_b)
    #     x_ba = self.gen_a.decode(c_b, s_a)
    #     x_ba_dis_out = self.dis_a(x_ba)
    #
    #     c_a, s_a_prime = self.gen_a.encode(x_a)
    #     x_ab = self.gen_b.decode(c_a, s_b)
    #     x_ab_dis_out = self.dis_b(x_ab)
    #     #
    #     self.info_cont_loss_a = self.compute_info_cont_loss(s_a, x_ba_dis_out)
    #     self.info_cont_loss_b = self.compute_info_cont_loss(s_b, x_ab_dis_out)
    #     #
    #     self.loss_enc_content = self.info_cont_loss_a + self.info_cont_loss_b
    #     self.loss_enc_content.backward()
    #     self.dec_opt.step()

    # def compute_info_cont_loss(self, style_code, outs_fake):
    #     loss = 0
    #     num_cont_code = self.num_con_c
    #     for it, (out_fake) in enumerate(outs_fake):
    #         q_mu = out_fake['mu']
    #         q_var = out_fake['var']
    #         info_noise = style_code[:, -num_cont_code:].view(-1, num_cont_code).squeeze().squeeze()
    #         # print(q_mu.size())
    #         # print(q_var.size())
    #         # print(info_noise.size())
    #         # print(num_cont_code)
    #         # exit()
    #         loss += self.criterionQ_con(info_noise, q_mu, q_var) * 0.1
    #     return loss

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

    def cla_digit_update(self, x_a_unlabeled, sample_a_limited, sample_b):
        x_a_limited, label_a_limited = sample_a_limited
        x_b, label_b = sample_b
        # print('cla_update')
        # print(x_a.device())
        # exit()
        self.cla_digit_opt.zero_grad()
        s_a = Variable(torch.randn(x_a_limited.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a_limited)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # print("c_a")
        # print(c_a.size())
        # exit()
        # decode (within domain)
        # x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        # x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        label_predict_c_a = self.content_digit_classifier(c_a)
        label_predict_c_a_recon = self.content_digit_classifier(c_a_recon)
        label_predict_c_b = self.content_digit_classifier(c_b)
        label_predict_c_b_recon = self.content_digit_classifier(c_b_recon)

        ### extract x_a_unlabeled
        c_a_unlabeled, s_a_unlabeled = self.gen_a.encode(x_a_unlabeled)
        x_a_unlabeled_recon = self.gen_a.decode(c_a_unlabeled, s_a_unlabeled)
        c_a_unlabeled_recon, _ = self.gen_a.encode(x_a_unlabeled_recon)

        ### consistency of content-label prediction on unlabeled samples from target domain
        label_predict_c_a_unlabeled = self.content_digit_classifier(c_a_unlabeled)
        label_predict_c_a_unlabeled_recon = self.content_digit_classifier(c_a_unlabeled_recon)
        self.loss_content_digit_classifier_c_a_and_c_a_recon = self.compute_content_classifier_two_predictions_loss(
            label_predict_c_a_unlabeled, label_predict_c_a_unlabeled_recon)

        ### content prediction loss on labeled samples (and their reconstructed samples) from target domain
        self.loss_content_digit_classifier_c_a = self.compute_content_classifier_loss(label_predict_c_a, label_a_limited)
        self.loss_content_digit_classifier_c_a_recon = self.compute_content_classifier_loss(label_predict_c_a_recon,
                                                                                            label_a_limited)

        ### the 3 losses above on samples from source domain which are all labeled
        self.loss_content_digit_classifier_c_b = self.compute_content_classifier_loss(label_predict_c_b, label_b)
        self.loss_content_digit_classifier_c_b_recon = self.compute_content_classifier_loss(label_predict_c_b_recon, label_b)
        self.loss_content_digit_classifier_c_b_and_c_b_recon = self.compute_content_classifier_two_predictions_loss(
            label_predict_c_b_recon, label_predict_c_b)

        self.loss_cla_digit_total = self.loss_content_digit_classifier_c_a + self.loss_content_digit_classifier_c_a_recon + \
                                    self.loss_content_digit_classifier_c_b + self.loss_content_digit_classifier_c_b_recon + \
                                    self.loss_content_digit_classifier_c_a_and_c_a_recon + \
                                    self.loss_content_digit_classifier_c_b_and_c_b_recon
        self.loss_cla_digit_total.backward()
        self.cla_digit_opt.step()

    def cla_domain_update(self, x_a, x_b):
        label_a = Variable(torch.zeros(x_a.size(0)).long().cuda())
        label_b = Variable(torch.ones(x_b.size(0)).long().cuda())
        # x_a_limited, label_a_limited = sample_a_limited
        # x_b, label_b = sample_b
        # print('cla_update')
        # print(x_a.device())
        # exit()
        self.cla_domain_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # print("c_a")
        # print(c_a.size())
        # exit()
        # decode (within domain)
        # x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        # x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        label_predict_c_a = self.content_domain_classifier(c_a)
        label_predict_c_a_recon = self.content_domain_classifier(c_a_recon)
        label_predict_c_b = self.content_domain_classifier(c_b)
        label_predict_c_b_recon = self.content_domain_classifier(c_b_recon)

        ### content prediction loss on a (and their reconstructed samples) from target domain
        self.loss_content_domain_classifier_c_a = self.compute_content_classifier_loss(label_predict_c_a, label_a)
        self.loss_content_domain_classifier_c_a_recon = self.compute_content_classifier_loss(label_predict_c_a_recon, label_a)
        self.loss_content_domain_classifier_c_a_and_c_a_recon = self.compute_content_classifier_two_predictions_loss(
            label_predict_c_a_recon, label_predict_c_a)

        ### the 3 losses above on samples from source domain which are all labeled
        self.loss_content_domain_classifier_c_b = self.compute_content_classifier_loss(label_predict_c_b, label_b)
        self.loss_content_domain_classifier_c_b_recon = self.compute_content_classifier_loss(label_predict_c_b_recon, label_b)
        self.loss_content_domain_classifier_c_b_and_c_b_recon = self.compute_content_classifier_two_predictions_loss(
            label_predict_c_b_recon, label_predict_c_b)

        self.loss_cla_domain_total = self.loss_content_domain_classifier_c_a + self.loss_content_domain_classifier_c_a_recon + \
                                    self.loss_content_domain_classifier_c_b + self.loss_content_domain_classifier_c_b_recon + \
                                    self.loss_content_domain_classifier_c_a_and_c_a_recon + \
                                    self.loss_content_domain_classifier_c_b_and_c_b_recon
        self.loss_cla_domain_total.backward()
        self.cla_domain_opt.step()

    def cla_inference(self, test_loader_a, test_loader_b):
        accu_content_classifier_c_a = []
        accu_content_classifier_c_a_recon = []
        accu_content_classifier_c_b = []
        accu_content_classifier_c_b_recon = []
        for it_inf, (samples_a_test, samples_b_test) in enumerate(zip(test_loader_a, test_loader_b)):
            x_a, label_a = samples_a_test[0].cuda().detach(), samples_a_test[1].cuda().detach()
            x_b, label_b = samples_b_test[0].cuda().detach(), samples_b_test[1].cuda().detach()

            s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

            # encode
            c_a, s_a_prime = self.gen_a.encode(x_a)
            c_b, s_b_prime = self.gen_b.encode(x_b)
            # print("c_a")
            # print(c_a.size())
            # exit()
            # decode (within domain)
            # x_a_recon = self.gen_a.decode(c_a, s_a_prime)
            # x_b_recon = self.gen_b.decode(c_b, s_b_prime)
            # decode (cross domain)
            x_ba = self.gen_a.decode(c_b, s_a)
            x_ab = self.gen_b.decode(c_a, s_b)
            # encode again
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
            c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

            label_predict_c_a = self.content_digit_classifier(c_a)
            label_predict_c_a_recon = self.content_digit_classifier(c_a_recon)
            label_predict_c_b = self.content_digit_classifier(c_b)
            label_predict_c_b_recon = self.content_digit_classifier(c_b_recon)

            # self.loss_content_classifier_c_a = self.compute_content_classifier_loss(label_predict_c_a, label_a)
            # self.loss_content_classifier_c_a_recon = self.compute_content_classifier_loss(label_predict_c_a_recon, label_a)
            # self.loss_content_classifier_c_a_and_c_a_recon = self.compute_content_classifier_two_predictions_loss(label_predict_c_a_recon, label_predict_c_a)
            #
            # self.loss_content_classifier_b = self.compute_content_classifier_loss(label_predict_c_b, label_b)
            # self.loss_content_classifier_c_b_recon = self.compute_content_classifier_loss(label_predict_c_b_recon, label_b)
            # self.loss_content_classifier_c_b_and_c_b_recon = self.compute_content_classifier_two_predictions_loss(label_predict_c_b_recon,
            #                                                                                  label_predict_c_b)

            accu_content_classifier_c_a.append(self.compute_content_classifier_accuracy(label_predict_c_a, label_a))
            accu_content_classifier_c_a_recon.append(
                self.compute_content_classifier_accuracy(label_predict_c_a_recon, label_a))
            accu_content_classifier_c_b.append(self.compute_content_classifier_accuracy(label_predict_c_b, label_b))
            accu_content_classifier_c_b_recon.append(
                self.compute_content_classifier_accuracy(label_predict_c_b_recon, label_b))

        self.accu_content_classifier_c_a = self.mean_list(accu_content_classifier_c_a)
        self.accu_content_classifier_c_a_recon = self.mean_list(accu_content_classifier_c_a_recon)
        self.accu_content_classifier_c_b = self.mean_list(accu_content_classifier_c_b)
        self.accu_content_classifier_c_b_recon = self.mean_list(accu_content_classifier_c_b_recon)

        self.accu_CC_all = self.mean_list([
            self.accu_content_classifier_c_a,
            self.accu_content_classifier_c_a_recon,
            self.accu_content_classifier_c_b,
            self.accu_content_classifier_c_b_recon
        ])

        # self.loss_cla_total = self.loss_content_classifier_c_a + self.loss_content_classifier_c_a_recon + \
        #                       self.loss_content_classifier_b + self.loss_content_classifier_c_b_recon + \
        #                       self.loss_content_classifier_c_a_and_c_a_recon + \
        #                       self.loss_content_classifier_c_b_and_c_b_recon
        # self.loss_cla_total.backward()
        # self.cla_opt.step()

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
        # print(x_ba.detach().size())
        # print(x_a.size())
        # exit()
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

    def compute_content_classifier_accuracy(self, label_predict, label_true, custom_batch_size_val=None):
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
        if custom_batch_size_val is None:
            accuracy = float(total_correct) / float(self.batch_size_val)
        else:
            accuracy = float(total_correct) / float(custom_batch_size_val)
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
        if self.enc_style_scheduler is not None:
            self.enc_style_scheduler.step()
        if self.enc_content_scheduler is not None:
            self.enc_content_scheduler.step()
        if self.dec_scheduler is not None:
            self.dec_scheduler.step()

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
        # Load content classifier
        last_model_name = get_model_list(checkpoint_dir, "con_cla")
        state_dict = torch.load(last_model_name)
        self.content_digit_classifier.load_state_dict(state_dict['con_cla'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.cla_digit_opt.load_state_dict(state_dict['con_cla'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.enc_style_scheduler = get_scheduler(self.enc_style_opt, hyperparameters, iterations)
        self.enc_content_scheduler = get_scheduler(self.enc_content_opt, hyperparameters, iterations)
        self.dec_scheduler = get_scheduler(self.dec_opt, hyperparameters, iterations)
        self.cla_digit_scheduler = get_scheduler(self.cla_digit_opt, hyperparameters, iterations)
        self.cla_domain_scheduler = get_scheduler(self.cla_domain_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        con_cla_name = os.path.join(snapshot_dir, 'con_cla_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')

        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'con_cla': self.content_digit_classifier.state_dict()}, con_cla_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict(),
                    'con_cla': self.cla_digit_opt.state_dict()}, opt_name)
