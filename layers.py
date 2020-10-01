# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

#########################
# FullyConnected Layers #
#########################
def FullyConnected(in_f, out_f):
    '''
        Fully connected layers used by both G and D
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.LeakyReLU(0.5)
    )

def G_out(in_f, out_f):
    '''
        Output layer of the generator model
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Tanh()
    )

def G_out_no_actv(in_f, out_f):
    '''
        Output layer of the ewm generator model without tanh()
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
    )

def D_out(in_f, out_f):
    '''
        Output layer of the discriminator model
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Sigmoid()
    )

########################
# Convolutional Layers #
########################
def ConvBlock(in_f, out_f):
    '''
        Convolutional blocks increase the depth of the feature maps from
        in_f -> out_f. The MaxPool2d funtion then reduces the feature
        map dimension by a factor of 2.
    '''
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, kernel_size = 3, padding = 1),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(2,2)
    )

def ConvBlock_BN(in_f, out_f):
    '''
        Convolutional blocks increase the depth of the feature maps from
        in_f -> out_f. The MaxPool2d funtion then reduces the feature
        map dimension by a factor of 2.
    '''
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, kernel_size = 3, padding = 1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(out_f),
        nn.MaxPool2d(2,2)
    )

def DeconvBlock(in_f, out_f):
    '''
        Deconvolution function that replaces the usual convolutional transpose
        operation with two linear operations - a bilinear upsample and a
        standard convolution. We set the stride of the convolution operation to
        1 in order to maintain the image dimension after upsampling.
    '''
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(0.2)
    )

def DeconvBlockLast(in_f, out_f):
    '''
        ConvTranspose blocks decrease the depth of the feature maps from
        in_f -> out_f.
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(in_f, out_f, 2, stride = 2),
        nn.Tanh()
    )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

###################
# Residual Layers #
###################
class DoubleRes(nn.Module):
    def __init__(self, Block, inplanes, planes, stride=1):
        super(DoubleRes, self).__init__()
        self.res1 = Block(inplanes, planes, stride)
        self.res2 = Block(planes, planes, 1)

    def forward(self, x):
        #print(x.shape)
        out = self.res1(x)
        out = self.res2(out)
        #print(x.shape)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

        self.bypass = None
        if inplanes != planes or stride > 1:
            self.bypass = nn.Conv2d(inplanes, planes, kernel_size=1,
                                    stride=stride, padding=0, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.bypass is not None:
            outbp = self.bypass(x)
            out += outbp
        else:
            out += residual
            
        out = self.relu(out)
        return out


class ConvTransposeLayer(nn.Module):
    def __init__(self, deconv_inplanes, deconv_outplanes, res_outplanes):
        super(ConvTransposeLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(deconv_inplanes,
                                         deconv_outplanes, kernel_size=4,
                                         stride=2, padding=1, bias=False)
        self.res = DoubleRes(BasicBlock, deconv_outplanes,
                             res_outplanes, stride=1)

    def forward(self, x):
        out = self.deconv(x)
        out = self.res(out)
        return out
