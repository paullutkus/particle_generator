# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import math

from layers import *

class Var_ResEncoder(nn.Module): 
    
    def __init__(self, l_dim, depth, strides, inplanes=4, input_channels=1):
        self.inplanes = inplanes
        super(Var_ResEncoder, self).__init__()
        
        self.res_enc_blocks = nn.Sequential(*[DoubleRes(BasicBlock, in_f, out_f, stride) for in_f, out_f, stride
                                        in zip(depth, depth[1:], strides)])
        
        #self.avgpool = nn.AvgPool2d(7, stride=1, padding=3)
        #self.fc = nn.Linear(1024, l_dim)
        #self.fc.weight.data.copy_(torch.eye(l_dim, 1024))
        #self.relu = nn.ReLU(inplace=True)

        self.mu_fc = nn.Linear(1568, l_dim)
        self.logvar_fc = nn.Linear(1568, l_dim)
        self.mu_fc.weight.data.copy_(torch.eye(l_dim, 1568))
        self.logvar_fc.weight.data.copy_(torch.eye(l_dim, 1568))

    def _make_encoding_layer(self, inplanes, planes, stride=2):
        return DoubleRes(BasicBlock, inplanes, planes, stride=stride)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        x = self.res_enc_blocks(x)
        
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        #x = self.relu(x)

        return self.mu_fc(x), self.logvar_fc(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ResDecoder(nn.Module):
    
    def __init__(self, l_dim, depth, inplanes=4, input_channels=1):
        super(ResDecoder, self).__init__()
        self.inplanes = inplanes
        self.fc = nn.Linear(l_dim, 1568)
        self.fc.weight.data.copy_(torch.eye(1568, l_dim))
        self.relu = nn.ReLU(inplace=True)
        #self.sigmoid = nn.Sigmoid()

        self.res_dec_blocks = nn.Sequential(*[ConvTransposeLayer(in_f, out_f, out_f) for in_f, out_f
                                            in zip(depth, depth[1:])])

        self.double_res = DoubleRes(BasicBlock, depth[-1], 
                                input_channels, stride=1)
         
        self.conv = nn.Conv2d(input_channels, input_channels,
                                kernel_size=3, stride=1,
                                padding=1)
         
    def _make_decoding_layer(self, inplanes, deconvplanes, resnetplanes):
        return ConvTransposeLayer(inplanes, deconvplanes, resnetplanes) 

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = torch.reshape(x, (-1, 32, 7, 7))
        x = self.res_dec_blocks(x)
        x = self.double_res(x) 
        x = self.conv(x)

        return self.sigmoid(x)

class Var_ResAutoEncoder(nn.Module):
    '''
        Convolutional AutoEncoder model.
        This model assumes the use of 1-channel images. If using 3-channels,
        modify each instance of [1] to [3].
    '''
    def __init__(self, enc_depth, dec_depth, strides, l_dim):
        super().__init__()
        self.l_dim        = l_dim                # Computed in setup_model.py
        self.enc_features = enc_depth            # [1] + [4, 8, 16, 32]
        self.dec_features = dec_depth  # [l_dim] + [32, 16, 8] + [1]
        self.input_channels = 1
        self.inplanes = 4
        self.strides = strides
        self.encoder = Var_ResEncoder(self.l_dim, self.enc_features,
                                   self.strides, self.inplanes,
                                   self.input_channels) 
        self.decoder = ResDecoder(self.l_dim, self.dec_features, self.inplanes,
                                   self.input_channels) 

    def forward(self, x):
        x, mu, logvar = self.encoder(x)
        x = self.decoder(x)
        return x, mu, logvar
