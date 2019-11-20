################################################################################
# ae.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.23.2019
# Purpose: - This file provides the PyTorch versions of the encoder and
#            decoder network models for use in the particle generator
#          - Model architecture is decided at run-time based on the
#            provided command line arguments.
################################################################################

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

def fc_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.ReLU(True)
    )

class Encoder(nn.Module):
    def __init__(self, enc_sizes):
        super().__init__()
        self.fc_blocks = nn.Sequential(*[fc_block(in_f, out_f) for in_f, out_f
                                        in zip(enc_sizes, enc_sizes[1:])])
        self.last = nn.Linear(enc_sizes[-2], enc_sizes[-1])

    # Initialize the weights
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        return self.last(self.fc_blocks(x))

class Decoder(nn.Module):
    def __init__(self, dec_sizes):
        super().__init()
        self.fc_blocks = nn.Sequential(*[fc_block(in_f, out_f) for in_f, out_f
                                        in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Sequential(nn.Linear(dec_sizes[-2], dec_sizes[-1]), nn.Tanh())

    # Initialize the weights
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        return self.last(self.fc_blocks(x))

class AutoEncoder(nn.Module):
    def __init__(self, enc_sizes, dec_sizes):
        super().__init__()
        self.enc_sizes = [*enc_sizes]
        self.dec_sizes = [*dec_sizes]

        self.encoder = Encoder(self.enc_sizes)
        self.decoder = Decoder(self.dec_sizes)

    def weights_init(self):
        self.encoder.weights_init()
        self.decoder.weights_init()

    def forward(self, x):
        return self.decoder(self.encoder(x))
