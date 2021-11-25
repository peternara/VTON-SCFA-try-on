import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU()]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        
        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)

class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)
    
    
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h*w)
        feature_B = feature_B.view(b, c, h*w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_dim=6):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, False),
        )
        self.linear = nn.Linear(64 * 4 * 3, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x



class FlowDeCoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, use_bias=True):
        super(FlowDeCoder, self).__init__()

        self.uprelu = nn.ReLU()

        self.upconv6 = nn.ConvTranspose2d(input_nc, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm6 = nn.BatchNorm2d(ngf*8)

        #here input channel is doubled because of soft connection
        self.upconv5 = nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm5 = nn.BatchNorm2d(ngf*8)
        
        self.upconv4 = nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm4 = nn.BatchNorm2d(ngf*4)

        self.upconv3 = nn.ConvTranspose2d(ngf*4, ngf*4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm3 = nn.BatchNorm2d(ngf*4)

        self.upconv2 = nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upnorm2 = nn.BatchNorm2d(ngf*2)
        
        self.upconv1 = nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)


    def forward(self, input):
        x6 = input
        x5 = self.operation(x6, self.uprelu, self.upconv6, self.upnorm6)
        x4 = self.operation(x5, self.uprelu, self.upconv5, self.upnorm5)
        x3 = self.operation(x4, self.uprelu, self.upconv4, self.upnorm4)
        x2 = self.operation(x3, self.uprelu, self.upconv3, self.upnorm3)
        x1 = self.operation(x2, self.uprelu, self.upconv2, self.upnorm2)


        x0 = self.uprelu(x1)
        out = self.upconv1(x0)

        return out 


    def operation(self, input, activation, conv, norm):
        x = activation(input)
        x = conv(x)
        x = norm(x)

        return x


class AFGAN(nn.Module):
    def __init__(self, inpur_An=3, input_Bn=18, output_nc=2, ngf=64, use_bias=True):
        super(AFGAN, self).__init__()
        self.extractionA = FeatureExtraction(inpur_An, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d)
        self.extractionB = FeatureExtraction(input_Bn, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.fr = FeatureRegression(input_nc=192, output_dim=3072)
        self.flow = FlowDeCoder(input_nc=256, output_nc=output_nc, ngf=ngf, use_bias=use_bias)

    
    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB) #([6, 192, 16, 12])
        regression = self.fr(correlation)
        regression = regression.reshape(regression.size(0), -1, 4, 3)
        sin_flow = self.flow(regression)
        return sin_flow


def flow_warping(input, flow_field):
    [b,_,h,w] = flow_field.size()

    source_copy = F.interpolate(input, (h,w)) 

    x = torch.arange(w).view(1, -1).expand(h, -1).float()
    y = torch.arange(h).view(-1, 1).expand(-1, w).float()
    x = 2*x/(w-1)-1
    y = 2*y/(h-1)-1
    grid = torch.stack([x,y], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    flow_x = (2*flow_field[:,0,:,:]/(w-1)).view(b,1,h,w)
    flow_y = (2*flow_field[:,1,:,:]/(h-1)).view(b,1,h,w)
    
    flow = torch.cat((flow_x, flow_y), 1)
    flow_x = flow.clone()
    final_grid = (grid + flow_x).permute(0, 2, 3, 1)
    warp = F.grid_sample(source_copy, final_grid)
    return warp, final_grid   

