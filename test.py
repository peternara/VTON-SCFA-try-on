# coding=utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn
import os.path as osp
import os
import warnings
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from visualization import load_checkpoint, save_images
from dataReader import DGDataset, DGDataLoader
from config import parser
from models.networks import AttU_Net, GLSP
from flownet import AFGAN, flow_warping
import torch.nn.functional as F
torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def testSGM(opt, test_loader, model):
    print('----Testing of module {} started----'.format(opt.name))
    model.to(device)
    model.eval()

    length = len(test_loader.data_loader)
    step = 0
    pbar = tqdm(total=length)

    inputs = test_loader.next_batch()
    while inputs is not None:
        im_name = inputs['im_name']
        c = inputs['cloth'].to(device)
        point_line = inputs['point_line'].to(device)
        can_bin = (inputs['can_bin'].unsqueeze(1)).to(device)
        ptexttp = (inputs['ptexttp'].unsqueeze(1)).to(device)
        im_parse = inputs['parse_model'].to(device)

        input_concat = torch.cat([can_bin, im_parse.unsqueeze(1)*ptexttp, point_line, c], 1)
        output_segm = model(input_concat)

        """
        0 -> Background 1 -> Hair 4 -> Upclothes 5 -> Left-shoe  6 -> Right-shoe 7 -> Noise 
        8 -> Pants 9 -> Left_leg 10 -> Right_leg 11 -> Left_arm 12 -> Face 13 -> Right_arm
        """
        save_images(output_segm[:,4,:,:], im_name, osp.join(opt.save_dir, opt.datamode, 'SemanticMap'))

        inputs = test_loader.next_batch()
        step+=1
        pbar.update(1)


def testGAM(opt, test_loader, model):
    print('----Testing of module {} started----'.format(opt.name))
    model.to(device)
    model.eval()
    
    glsp = GLSP(8, 21, 0, 32)
    load_checkpoint(glsp, os.path.join(opt.checkpoint_dir, 'SGM', 'sgm_final.pth'))
    glsp.to(device)
    glsp.eval()
    

    length = len(test_loader.data_loader)
    step = 0
    pbar = tqdm(total=length)

    inputs = test_loader.next_batch()
    while inputs is not None:
        im_name = inputs['im_name']
        c = inputs['cloth'].to(device)
        can_bin = (inputs['can_bin'].unsqueeze(1)).to(device)
        ptexttp = (inputs['ptexttp'].unsqueeze(1)).to(device)
        point_line = inputs['point_line'].to(device)
        im_parse = inputs['parse_model'].to(device)

        input_concat = torch.cat([can_bin, im_parse*ptexttp, point_line, c], 1)
        output_segm = glsp(input_concat)
        output_segm = F.log_softmax(output_segm, dim=1)
        output_argm = torch.max(output_segm, dim=1, keepdim=True)[1]
        final_segm = torch.zeros(output_segm.shape).to(device).scatter(1, output_argm, 1.0)

        sin_flow = model(c, final_segm[:,4,:,:].unsqueeze(1))
        warped_garment, _  = flow_warping(c, sin_flow)

        save_images(warped_garment, im_name, osp.join(opt.save_dir, opt.datamode, 'WarpedGarment'))

        inputs = test_loader.next_batch()
        step+=1
        pbar.update(1)



def testTSM(opt, test_loader, model):
    print('----Testing of module {} started----'.format(opt.name))
    model.to(device)
    model.eval()


    glsp = GLSP(6, 21, 3, 32)
    load_checkpoint(glsp, os.path.join(opt.checkpoint_dir, 'SGM', 'sgm_final.pth'))
    glsp.to(device)
    glsp.eval()

    afgan = AFGAN(inpur_An=3, input_Bn=1, ngf=64, use_bias=True)
    load_checkpoint(afgan, os.path.join(opt.checkpoint_dir, 'AFGAN', 'gam_final.pth'))
    afgan.to(device)
    afgan.eval()

    length = len(test_loader.data_loader)
    step = 0
    pbar = tqdm(total=length)

    inputs = test_loader.next_batch()
    while inputs is not None:
        im_name = inputs['im_name']
        c = inputs['cloth'].to(device)
        im_ttp = inputs['texture_t_prior'].to(device)
        im_a = inputs['im_a'].to(device)
        can_bin = (inputs['can_bin'].unsqueeze(1)).to(device)
        point_line = inputs['point_line'].to(device)
        ptexttp = (inputs['ptexttp'].unsqueeze(1)).to(device)
        im_parse = inputs['parse_model'].to(device)
        cw = inputs['warp_cloth'].to(device)
        
        input_concat = torch.cat([can_bin, im_parse*ptexttp, point_line, c], 1)
        output_segm = glsp(input_concat)
        output_segm = F.log_softmax(output_segm, dim=1)
        output_argm = torch.max(output_segm, dim=1, keepdim=True)[1]
        final_segm = torch.zeros(output_segm.shape).to(device).scatter(1, output_argm, 1.0)
       
        im_cm = final_segm[:,4,:,:].unsqueeze(1)
        sin_flow = afgan(c, im_cm)
        sample_im_s, _ = flow_warping(c, sin_flow)
        print(im_cm.shape)
        cw = im_cm*sample_im_s

        arm_par = final_segm[:,11,:,:].unsqueeze(1) + final_segm[:,13,:,:].unsqueeze(1)
        input_tom = torch.cat((final_segm, im_ttp, cw, im_a*arm_par), 1)
        print(input_tom.shape)
        output_tom = model(input_tom)
        

        save_images(output_tom, im_name, osp.join(opt.save_dir, opt.datamode, 'TryOnResult'))

        inputs = test_loader.next_batch()
        step+=1
        pbar.update(1)


def main():
    opt = parser()

    test_dataset = DGDataset(opt)
    # create dataloader
    test_loader = DGDataLoader(opt, test_dataset)

    if opt.name == 'SGM':
    
        model = GLSP(6, 21, 3, 32)
        checkpoint_path = osp.join(opt.checkpoint_dir, opt.name, 'sgm_final.pth')
        load_checkpoint(model, checkpoint_path)
        testSGM(opt, test_loader, model)
   
    elif opt.name == 'GAM':

        model = AFGAN(inpur_An=3, input_Bn=1, ngf=64, use_bias=True)
        checkpoint_path = osp.join(opt.checkpoint_dir, opt.name, 'gam_final.pth')
        load_checkpoint(model, checkpoint_path)
        testGAM(opt, test_loader, model)

    elif opt.name == 'TSM':

        model = AttU_Net(30, 3)
        checkpoint_path = osp.join(opt.checkpoint_dir, opt.name, 'tsm_final.pth')
        load_checkpoint(model, checkpoint_path)
        testTSM(opt, test_loader, model)
      
if __name__ == '__main__':
    main()
    
