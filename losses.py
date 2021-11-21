import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from config import parser
import numpy as np
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class criterion_vgg(nn.Module):
    def __init__(self):
        super(criterion_vgg, self).__init__()
    
    def forward(self, w, x, y):  
        abs_diff = torch.abs(x - y)
        l1 = torch.mean(abs_diff, dim=[1,2,3]).unsqueeze(1)
        l1 = w*l1
        return l1
    
    
class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids
        
        self.loss_l1 = criterion_vgg()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
            
        loss_f0 = self.loss_l1(self.weights[self.layids[0]], x_vgg[self.layids[0]], y_vgg[self.layids[0]].detach())
        loss_f1 = self.loss_l1(self.weights[self.layids[1]], x_vgg[self.layids[1]], y_vgg[self.layids[1]].detach())
        loss_f2 = self.loss_l1(self.weights[self.layids[2]], x_vgg[self.layids[2]], y_vgg[self.layids[2]].detach())
        loss_f3 = self.loss_l1(self.weights[self.layids[3]], x_vgg[self.layids[3]], y_vgg[self.layids[3]].detach())
        loss_f4 = self.loss_l1(self.weights[self.layids[4]], x_vgg[self.layids[4]], y_vgg[self.layids[4]].detach())
        
        loss = torch.cat([loss_f0, loss_f1, loss_f2, loss_f3, loss_f4], 1)
        return loss

    def warp(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        loss += self.weights[4] * self.criterion(x_vgg[4], y_vgg[4].detach())
        return loss


class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, x2):
        dt = torch.abs(x1 - x2)
        return dt


class DT2(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) +
                        torch.mul(y1 - y2, y1 - y2))
        return dt


class GicLoss(nn.Module):
    def __init__(self, opt):
        super(GicLoss, self).__init__()
        self.dT = DT()
        self.opt = opt

    def forward(self, grid):
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]
        Gxcenter = Gx[:, 1:self.opt.fine_height - 1, 1:self.opt.fine_width - 1]
        Gxup = Gx[:, 0:self.opt.fine_height - 2, 1:self.opt.fine_width - 1]
        Gxdown = Gx[:, 2:self.opt.fine_height, 1:self.opt.fine_width - 1]
        Gxleft = Gx[:, 1:self.opt.fine_height - 1, 0:self.opt.fine_width - 2]
        Gxright = Gx[:, 1:self.opt.fine_height - 1, 2:self.opt.fine_width]

        Gycenter = Gy[:, 1:self.opt.fine_height - 1, 1:self.opt.fine_width - 1]
        Gyup = Gy[:, 0:self.opt.fine_height - 2, 1:self.opt.fine_width - 1]
        Gydown = Gy[:, 2:self.opt.fine_height, 1:self.opt.fine_width - 1]
        Gyleft = Gy[:, 1:self.opt.fine_height - 1, 0:self.opt.fine_width - 2]
        Gyright = Gy[:, 1:self.opt.fine_height - 1, 2:self.opt.fine_width]

        dtleft = self.dT(Gxleft, Gxcenter)
        dtright = self.dT(Gxright, Gxcenter)
        dtup = self.dT(Gyup, Gycenter)
        dtdown = self.dT(Gydown, Gycenter)

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))


# NEW LOSS
class NewL1Loss(nn.Module):
    def __init__(self):
        super(NewL1Loss, self).__init__()

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        max_diff = torch.max(diff)
        weight = diff / max_diff
        loss = weight * diff
        return loss.mean()


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class PixelSoftmaxLoss(nn.Module):
    def __init__(self, weight):
        super(PixelSoftmaxLoss, self).__init__()
        self.loss = CrossEntropyLoss(weight=weight)
    def forward(self, pred, target):
        pred = pred.reshape(pred.size(0), pred.size(1), -1) # batch, num_class, size, size
        _ , pos = torch.topk(target, 1, 1, True)
        pos = pos.reshape(pos.size(0), -1)
        loss = self.loss(pred, pos)
        return loss

def GANFeatLoss(num_D, pred_fake, pred_real):
    criterionFeat = nn.L1Loss()
    # GAN feature matching loss
    loss_G_GAN_Feat = 0

    feat_weights = 1.0
    D_weights = 1.0 / num_D
    for i in range(num_D):
        for j in range(len(pred_fake[i])-1):
            loss_G_GAN_Feat += D_weights * feat_weights * \
                criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
    return loss_G_GAN_Feat

####################################################################################################
####################################################################################################
def gmm_loss(grid, parse_cloth, warp_cloth, pwc, loss_warp):
    opt = parser()
    
    loss_l1 = nn.L1Loss()
    loss_vgg = VGGLoss()
    
    lambda0 = 1
    lambda1 = 1
    lambda2 = 1
    lambda3 = 40
    lambda4 = 0.5
    
    # loss for warped cloth
    ls0 = loss_l1(parse_cloth, warp_cloth)
    
    # grid regularization loss
    Lgic = GicLoss(opt)(grid)
    # 200x200 = 40.000 * 0.001
    Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])

    # total GMM loss
    # lambda0*ls0
    loss = lambda0*ls0 + lambda2*loss_warp  + lambda3*Lgic
    
    return loss


def segm_unet_loss(output, target):

    """
        0 -> Background 1 -> Hair 4 -> Upclothes 5 -> Left-shoe  6 -> Right-shoe 7 -> Noise 
        8 -> Pants 9 -> Left_leg 10 -> Right_leg 11 -> Left_arm 12 -> Face 13 -> Right_arm
    """
    l, m, h = 0.5, 1, 1.5
    weights = np.array([l, l, l, l, h, l, l, l, l, l, l, m, l, m, l, l, l, l, l, l, l], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    
    loss_ce = nn.CrossEntropyLoss(weight=weights)
    loss = loss_ce(output, target)

    return loss

def tom_loss(person_r, img_tryon, im, mask_c, im_c_mask):

    # (person_r, img_tryon, im, mask_c, im_c_mask)
    
    criterionFeat = nn.L1Loss()
    loss_vgg = VGGLoss()

    lambda0 = 1

    loss_cp = criterionFeat(person_r, im)
    loss_cp_perc = torch.mean(loss_vgg(person_r, im))

    loss_mask_c = criterionFeat(mask_c, im_c_mask)

    loss_p = criterionFeat(img_tryon, im)
    loss_perc = torch.mean(loss_vgg(img_tryon, im))

    l_tt = loss_p + loss_perc + loss_mask_c + loss_cp + loss_cp_perc

    return l_tt


# CX loss
class CXLoss(nn.Module):

    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCxHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''
        # NCHW
        # print(featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            # See the torch document for functional.conv2d
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX

def CX_Loss(I_g, I_t):
    cxLoss = CXLoss(sigma=0.5).to(device)
    vgg = VGG().to(device)
    lambda_cx = 0.5
    style_layer = ['r32', 'r42']
    vgg_style = vgg(I_t, style_layer)
    vgg_fake = vgg(I_g, style_layer)
    cx_style_loss = 0

    for i, val in enumerate(vgg_fake):
        cx_style_loss += cxLoss(vgg_style[i], vgg_fake[i])
    cx_style_loss *= lambda_cx

    return cx_style_loss

# VGG for CX Loss
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]