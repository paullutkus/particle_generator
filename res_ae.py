# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import math

from layers import *


class ResEncoder(nn.Module):

    def __init__(self, l_dim, inplanes=4, input_channels=1):
        self.inplanes = inplanes
        super(ResEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, self.inplanes,
                               kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_encoding_layer(self.inplanes,
                                                self.inplanes*2,
                                                stride=1)

        self.layer2 = self._make_encoding_layer(self.inplanes*2,
                                                self.inplanes*4,
                                                stride=2)
        self.layer3 = self._make_encoding_layer(self.inplanes*4,
                                                self.inplanes*8,
                                                stride=2)
        self.layer4 = self._make_encoding_layer(self.inplanes*8,
                                                self.inplanes*16,
                                                stride=2)
        
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=3)
        self.fc = nn.Linear(256000, l_dim)

    def _make_encoding_layer(self, inplanes, planes, stride=2):
        return DoubleRes(BasicBlock, inplanes, planes, stride=stride)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)
	
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = torch.flatten(x)
        x = self.fc(x)

        return x


class ResDecoder(nn.Module):
    
    def __init__(self, l_dim, inplanes=4, input_channels=1):
        super(ResDecoder, self).__init__()
        self.inplanes = inplanes
        self.fc = nn.Linear(l_dim, 256000)

        self.layer1 = self._make_decoding_layer(self.inplanes*16,
                                           self.inplanes*8,
                                           self.inplanes*8)
    
        self.layer2 = self._make_decoding_layer(self.inplanes*8,
                                           self.inplanes*4,
                                           self.inplanes*4)
        self.layer3 = self._make_decoding_layer(self.inplanes*4,
                                           self.inplanes*2,
                                           self.inplanes*2)
        self.layer4 = self._make_decoding_layer(self.inplanes*2,
                                           self.inplanes*1,
                                           self.inplanes*1)
        
        self.conv = nn.Conv2d(self.inplanes*1, input_channels,
                                kernel_size=3, stride=1,
                                padding=1)
        
    def _make_decoding_layer(self, inplanes, deconvplanes, resnetplanes):
        return ConvTransposeLayer(inplanes, deconvplanes, resnetplanes) 

    def forward(self, x):
        x = self.fc(x)
        x = torch.reshape(x, (250, 64, 4, 4))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv(x)
        return x


class ResAutoEncoder(nn.Module):
    '''
        Convolutional AutoEncoder model.
        This model assumes the use of 1-channel images. If using 3-channels,
        modify each instance of [1] to [3].
    '''
    def __init__(self, enc_depth, dec_depth, l_dim):
        super().__init__()
        self.l_dim        = l_dim                # Computed in setup_model.py
        self.enc_features = enc_depth            # [1] + [4, 8, 16, 32]
        self.dec_features = [l_dim] + dec_depth  # [l_dim] + [32, 16, 8] + [1]
        self.input_channels = 1
        self.inplanes = 4
        self.encoder = ResEncoder(self.l_dim, self.inplanes,
                                   self.input_channels)
        self.decoder = ResDecoder(self.l_dim, self.inplanes,
                                   self.input_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
