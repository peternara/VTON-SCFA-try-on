# coding=utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import os.path as osp
import os
import time
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from visualization import load_checkpoint, save_checkpoint
from dataReader import DGDataset, DGDataLoader
from unet import UnetGenerator
from losses import VGGLoss, GicLoss, segm_unet_loss
from config import parser
import numpy as np
from e_Unet import AttU_Net, Segm_Net
from flownet import FlowNet
torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_segm(opt, train_loader, model):
    print('----Traning of module {} started----'.format(opt.name))
    model.to(device)
    model.train()
        
    #used scheduler for training seg mask generation module
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_lambda = lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda)
    
    for step in range(opt.previous_step, opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        c = inputs['cloth'].to(device)
        point_line = inputs['point_line'].to(device)
        can_bin = (inputs['can_bin'].unsqueeze(1)).to(device)
        ptexttp = (inputs['ptexttp'].unsqueeze(1)).to(device)
        im_parse = inputs['parse_model'].to(device)

        input_concat = torch.cat([can_bin, im_parse.unsqueeze(1)*ptexttp, point_line, c], 1)
        output_segm = model(input_concat)

        ############### Backward Pass ####################
        im_parse = im_parse.type(torch.long)

        loss = segm_unet_loss(output_segm, im_parse)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_gmm(opt, train_loader, model):
    print('----Traning of module {} started----'.format(opt.name))
    model.to(device)
    model.train()
    
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    gicloss = GicLoss(opt)

    #used scheduler for training of gmm
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_lambda = lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda)
    
    for step in range(opt.previous_step, opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im_c = inputs['parse_cloth'].to(device)
        c = inputs['cloth'].to(device)
        seg_shape = inputs['seg_shape'].to(device)

        im_cm = seg_shape[:,4,:,:].unsqueeze(1)

        sin_flow = model(c, im_cm)
        sample_im_s, grid = flow_warp(c, sin_flow)

        warped_cloth = im_cm*sample_im_s
        
        ############### Backward Pass ####################
        # grid regularization loss
        # 200x200 = 40.000 * 0.001

        Lgic = gicloss(grid)
        Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])

        loss = criterionL1(warped_cloth, im_c) + torch.mean(criterionVGG(warped_cloth, im_c)) + 40 * Lgic

        # update discriminator weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model):
    print('----Traning of module {} started----'.format(opt.name))
    model.to(device)
    model.train()

    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()

    gmm = FlowNet(inpur_An=3, input_Bn=1, ngf=64, use_bias=True)
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, 'GMM', 'step_200000.pth'))
    gmm.to(device)
    gmm.eval()

    #scheduler is not used during training
    optimizer_G = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_lambda_G = lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda = lr_lambda_G)

    for step in range(opt.previous_step, opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
        
        im = inputs['image'].to(device)
        c = inputs['cloth'].to(device)
        im_ttp = inputs['texture_t_prior'].to(device)
        im_cm = inputs['im_cloth_mask'].to(device)
        seg_shape = inputs['seg_shape'].to(device)
        part_arms = inputs['part_arms'].to(device)


        with torch.no_grad():
            sin_flow = gmm(c, im_cm)
            sample_im_s, _ = flow_warp(c, sin_flow)
            cw = im_cm*sample_im_s
        
        arm_par = seg_shape[:,11,:,:].unsqueeze(1) + seg_shape[:,13,:,:].unsqueeze(1)
        input_tom = torch.cat((seg_shape, im_ttp, cw, part_arms*arm_par), 1)
        img_tryon = model(input_tom)

        # ############### Backward Pass ####################
        loss_l1 = criterionL1(img_tryon, im)
        loss_vgg = criterionVGG(img_tryon, im)
        loss_G = loss_l1 + loss_vgg

        # update discriminator weights
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss_G: %4f' % (step+1, t, loss_G.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def main():
    # opt = get_opt()
    opt = parser()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = DGDataset(opt)

    # create dataloader
    train_loader = DGDataLoader(opt, train_dataset)

    # create model & train & save the final checkpoint
    if opt.stage == 'SEG':
        model = Segm_Net(5, 21, 0, 32)

        if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name)):
            os.makedirs(os.path.join(opt.checkpoint_dir, opt.name))

        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_segm(opt, train_loader, model)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'seg_final.pth'))

    elif opt.stage == 'GMM':

        model = FlowNet(inpur_An=3, input_Bn=1, ngf=64, use_bias=True)
        if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name)):
            os.makedirs(os.path.join(opt.checkpoint_dir, opt.name))
            
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))

    elif opt.stage == 'TOM':
        model = AttU_Net(30, 3)
        if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name)):
            os.makedirs(os.path.join(opt.checkpoint_dir, opt.name))

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
