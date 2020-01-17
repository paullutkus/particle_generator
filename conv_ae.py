# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from ae_layers import *

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
    def __init__(self, depth):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f) for in_f, out_f
                                           in zip(depth, depth[1:])])
    def forward(self, x):
        return self.conv_blocks(x)

class ConvDecoder(nn.Module):
    '''
        Convolutional AutoEncoder - Decoder branch
        args: depth (int list): list of integer feature depth sizes
                                that define the output volume of the
                                individual transposed convolution operations.
                                See ConvEncoder for example of list, which is
                                reversed in the Decoder model.
    '''
    def __init__(self, depth):
        super().__init__()
        self.deconv_blocks = nn.Sequential(*[deconv_blocks(in_f, out_f) for in_f, out_f
                                              in zip(depth, depth[1:])])
        self.activation = nn.Tanh()
    def forward(self, x):
        return self.activation(self.deconv_blocks(x))

class ConvAutoEncoder(nn.Module):
    '''
        Convolutional AutoEncoder model.
        This model assumes the use of 1-channel images. If using 3-channels,
        modify each instance of [1] to [3].
    '''
    def __init__(self, enc_depth, dec_depth):      # Examples:
        self.enc_features = [1] + enc_depth        # [1, 4, 8, 16, 32]
        self.dec_features = dec_depth + [1]        # [32, 16, 8, 4, 1]
        self.encoder = ConvEncoder(self.enc_features)
        self.decoder = ConvDecoder(self.dec_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
