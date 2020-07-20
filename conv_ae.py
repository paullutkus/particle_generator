# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from layers import *


class ConvEncoder(nn.Module):
    '''
        Convolutional AutoEncoder - Encoder branch
        args: depth (int list): list of integer feature depth sizes
                                that define the output volume of the
                                individual convolution operations.
                                The depth list is computed by dividing
                                the desired depth by the number of layers.
                                Ex: D = 32, n_layers = 4, depth = [4, 8, 16, 32]
                                Adding the initial channel: [1, 4, 8, 16, 32]
    '''
    def __init__(self, depth, l_dim):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[ConvBlock_BN(in_f, out_f) for in_f, out_f
                                           in zip(depth, depth[1:])])
        
        self.fc = nn.Linear(64*4*4, l_dim)
        self.fc.weight.data.copy_(torch.eye(l_dim, 64*4*4))
        self.relu = nn.ReLU(inplace=True)
        # self.last = nn.Conv2d(depth[-1], depth[-1]*2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_blocks(x)
        # x = self.last(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = self.relu(x)
        return x
        

class ConvDecoder(nn.Module):
    '''
        Convolutional AutoEncoder - Decoder branch
        args: depth (int list): list of integer feature depth sizes
                                that define the output volume of the
                                individual transposed convolution operations.
                                See ConvEncoder for example of list, which is
                                reversed in the Decoder model.
    '''
    def __init__(self, depth, l_dim):
        super().__init__()
        self.fc = nn.Linear(l_dim, 64*4*4)
        self.fc.weight.data.copy_(torch.eye(64*4*4, l_dim))
        self.relu = nn.ReLU(inplace=True)
        # self.first = nn.Conv2d(depth[-1] * 2, depth[-1], kernel_size=3, padding=1) 
        self.deconv_blocks = nn.Sequential(*[DeconvBlock(in_f, out_f) for in_f, out_f
                                              in zip(depth, depth[1:])])
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        # print(x.shape)
        x = torch.reshape(x, (-1, 64, 4, 4))
        # print(x.shape)
        # x = first(x)
        # print(x.shape)
        x = self.activation(self.deconv_blocks(x))
        return x


class ConvAutoEncoder(nn.Module):
    '''
        Convolutional AutoEncoder model.
        This model assumes the use of 1-channel images. If using 3-channels,
        modify each instance of [1] to [3].
    '''
    def __init__(self, enc_depth, dec_depth, l_dim):
        super().__init__()
        self.l_dim        = l_dim                # Computed in setup_model.py
        self.enc_features = enc_depth            # [1] + [4, 8, 16, 32]
        self.dec_features = dec_depth # [l_dim] + dec_depth  # [l_dim] + [32, 16, 8] + [1]
        self.encoder = ConvEncoder(self.enc_features, self.l_dim)
        self.decoder = ConvDecoder(self.dec_features, self.l_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
