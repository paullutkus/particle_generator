###############################################################################
# train_fns.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.15.2019
# Purpose: - This file provides training functions for both the GAN model
#            and the AE model. The desired training funcion is chosen at
#            runtime.
###############################################################################

# Imports
# Sys Imports
import os
import time
import errno
import shutil
from   datetime import datetime

# Python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from   statistics import mean

# Torch
import torch
import torch.nn as nn
import torchvision.utils
from   torch.nn         import init
from   torch.utils.data import DataLoader
from   torchvision      import transforms
from   torchvision      import datasets as dset

# My Stuff
import utils

def get_train_loop(config):
    '''
        Function for selection of appropriate training loop
    '''
    if config['MNIST']:
        if config['model'] == 'gan':
            return MNIST_GAN
        else:
            return MNIST_AE
    else:
        if config['model'] == 'gan':
            return LARCV_GAN
        else:
            return LARCV_AE

############################
#  Training Loops - MNIST  #
############################
def MNIST_GAN(epoch, epoch_start, G, G_optim, D, D_optim, dataloader, train_fn,
              history, best_stat, times, config, z_fixed):
    '''
        MNIST dataset training loop for GAN model. Used to train GAN as a
        proof-of-concept, i.e. that the linear GAN model is able to reproduce
        MNIST data, and is therefore (possibly) suited to reproducing the
        LArCV1 dataset. Hopefully this will extend to other datasets...
        - Args: G (Torch model): Generator model
                G_optim (function): G optimizer (either adam or sgd)
                D (Torch model): Discriminator model
                D_optim (function): D optimizer (either adam or sgd)
                Dataloader (iterable): Torch dataloader object wrapped as
                                       tqdm progress bar for terminal output
                train_fn (function): GAN training function selected in train.py
                history, best_stat, times, config (dicts): dictionaries
                epoch, epoch_start (ints)
                z_fixed (Torch tensor): Fixed vector for sampling G at the
                                        end of a training epoch
    '''
    for itr, (x, _) in enumerate(dataloader):
        tr_loop_start = time.time()

        metrics = train_fn(x)
        history, best_stat, best = utils.train_logger(history, best_stat, metrics)

        # Save checkpoint periodically
        if (itr % 2000 == 0):
            # G Checkpoint
            chkpt_G = utils.get_checkpoint(itr, epoch, G, G_optim)
            utils.save_checkpoint(chkpt_G, best, 'G', config['weights_save'])

            # D Checkpoint
            chkpt_D = utils.get_checkpoint(itr, epoch, D, D_optim)
            utils.save_checkpoint(chkpt_D, best, 'D', config['weights_save'])

        # Save Generator output periodically
        if (itr % 1000 == 0):
            z_rand = torch.randn(
                config['sample_size'], config['z_dim']).to(config['gpu'])
            sample = G(z_rand).view(-1, 1,
                       config['dataset'], config['dataset'])
            utils.save_sample(sample, epoch, itr, config['random_samples'])

        # Log the time at the end of training loop
        times['tr_loop_times'].append(time.time() - tr_loop_start)

    # Log the time at the end of the training epoch
    times['epoch_times'].append(time.time() - epoch_start)

    # Save Generator output using fixed vector at end of epoch
    sample = G(z_fixed).view(-1, 1, config['dataset'], config['dataset'])
    utils.save_sample(sample, epoch, itr, config['fixed_samples'])

    return history, best_stat, times

def MNIST_AE(epoch, epoch_start, AE, AE_optim, dataloader, train_fn, history,
             best_stat, times, config):
    '''
        MNIST dataset training loop for AE model. Used to train AE as a
        proof-of-concept.
        - Args: AE (Torch model): AutoEncoder model
                AR_optim (function): AE optimizer (either adam or sgd)
                Dataloader (iterable): Torch dataloader object wrapped as
                                       tqdm progress bar for terminal output
                train_fn (function): AE training function selected in train.py
                history, best_stat, times, config (dicts): dictionaries
                epoch, epoch_start (ints)
    '''
    for itr, (x, _) in enumerate(dataloader):
        tr_loop_start = time.time()

        metrics = train_fn(x, itr, epoch)
        history, best_stat = utils.train_logger(history, best_stat, metrics)

        # Log the time at the end of training loop
        times['tr_loop_times'].append(time.time() - tr_loop_start)

    # Log the time at the end of the training epoch
    times['epoch_times'].append(time.time() - epoch_start)

    return history, best_stat, times

############################
#  Training Loops - LARCV  #
############################
def LARCV_GAN(epoch, epoch_start, G, G_optim, D, D_optim, dataloader, train_fn,
              history, best_stat, times, config, z_fixed):
    '''
        LArCV dataset training loop for GAN model.
        - Args: G (Torch model): Generator model
                G_optim (function): G optimizer (either adam or sgd)
                D (Torch model): Discriminator model
                D_optim (function): D optimizer (either adam or sgd)
                Dataloader (iterable): Torch dataloader object wrapped up as
                                       tqdm progress bar for terminal output
                train_fn (function): GAN training function selected in train.py
                history, best_stat, times, config (dicts): dictionaries
                epoch, epoch_start (ints)
                z_fixed (Torch tensor): Fixed vector for sampling G at the
                                        end of a training epoch
    '''
    for itr, x in enumerate(dataloader):
            tr_loop_start = time.time()

            metrics = train_fn(x)
            history, best_stat, best = utils.train_logger(
                history, best_stat, metrics)

            # Save checkpoint periodically
            if (itr % 4000 == 0):
                # G Checkpoint
                chkpt_G = utils.get_checkpoint(itr, epoch, G, G_optim)
                utils.save_checkpoint(chkpt_G, best, 'G',
                                      config['weights_save'])

                # D Checkpoint
                chkpt_D = utils.get_checkpoint(itr, epoch, D, D_optim)
                utils.save_checkpoint(chkpt_D, best, 'D',
                                      config['weights_save'])

            # Save Generator output periodically
            if (itr % 1000 == 0):
                z_rand = torch.randn(config['sample_size'],
                                     config['z_dim']).to(config['gpu'])
                sample = G(z_rand).view(-1, 1,
                                        config['dataset'], config['dataset'])
                utils.save_sample(sample, epoch, itr, config['random_samples'])

            # Log the time at the end of training loop
            times['tr_loop_times'].append(time.time() - tr_loop_start)

    # Log the time at the end of the training epoch
    times['epoch_times'].append(time.time() - epoch_start)

    # Save Generator output using fixed vector at end of epoch
    sample = G(z_fixed).view(-1, 1, config['dataset'], config['dataset'])
    utils.save_sample(sample, epoch, itr, config['fixed_samples'])

    return history, best_stat, times

def LARCV_AE(epoch, epoch_start, AE, AE_optim, dataloader, train_fn, history,
             best_stat, times, config):
    '''
        LArCV dataset training loop for AE model.
        - Args: AE (Torch model): AutoEncoder model
                AR_optim (function): AE optimizer (either adam or sgd)
                Dataloader (iterable): Torch dataloader object wrapped as
                                       tqdm progress bar for terminal output
                train_fn (function): AE training function selected in train.py
                x_fixed: (torch tensor): Torch tensor assigned to a fixed
                                         variable at start of training. Used
                                         to visualize model progress/evolution.
                history, best_stat, times, config (dicts): dictionaries
                epoch, epoch_start (ints)
    '''
    for itr, x in enumerate(dataloader):
        tr_loop_start = time.time()

        metrics = train_fn(x, itr, epoch)
        history, best_stat = utils.train_logger(history, best_stat, metrics)

        # Log the time at the end of training loop
        times['tr_loop_times'].append(time.time() - tr_loop_start)

    # Log the time at the end of the training epoch
    times['epoch_times'].append(time.time() - epoch_start)

    return history, best_stat, times

########################
#  Training Functions  #
########################
def GAN_train_fn(G, D, G_optim, D_optim, loss_fn, config, G_D=None):
    '''
        GAN training function
        Does: Initializes the training function for the GAN model.
        Args: - G (Torch neural network): Generator model
              - D (Torch neural network): Discriminator model
              - G_D (Torch neural network, optional): Parallel model
              - z_fixed (Torch tensor): Fixed tensor of Gaussian noise for
                                        sampling Generator function.
              - loss_fn (function): Torch loss function
    '''
    def train(x):
        '''
            GAN training function
            Does: Trains the GAN model
            Args: x (Torch tensor): Real data input image
            Returns: list of training metrics
        '''
        # Make sure models are in training mode and that
        # optimizers are zeroed, just in case.
        G.train()
        D.train()
        D_optim.zero_grad()
        G_optim.zero_grad()

        # Move data to GPU (if not there already) and flatten into vector
        x = x.view(config['batch_size'], -1).to(config['gpu'])

        # Set up 'real' and 'fake' targets
        real_target = torch.ones(config['batch_size'], 1).to(config['gpu'])
        fake_target = torch.zeros(config['batch_size'], 1).to(config['gpu'])

        # Train D
        ## Generate fake data using G
        fake = torch.randn(config['batch_size'], config['z_dim']).to(config['gpu'])
        fake = G(fake).detach() # Detach so that gradients are not calc'd for G

        ## 1.1 Train D on real data
        real_pred  = D(x)
        real_error = loss_fn(real_pred, real_target) # Calculate BCE
        real_error.backward() # Backprop

        ## 1.2 Train D on fake data
        fake_pred  = D(fake)
        fake_error = loss_fn(fake_pred, fake_target) # Calculate BCE
        fake_error.backward() # Backprop

        ## Update D weights
        D_optim.step()

        # Train G
        ## Generate fake data and pass through D
        fake = torch.randn(config['batch_size'], config['z_dim']).to(config['gpu'])
        G_pred  = D(G(fake))
        G_error = loss_fn(G_pred, real_target) # Calculate BCE
        G_error.backward() # Backprop

        # Update G weights
        G_optim.step()

        # Return training metrics
        metrics = {'g_loss'     : float(G_error.item()),
                   'd_loss_real': float(real_error.item()),
                   'd_loss_fake': float(fake_error.item())}
        return metrics
    return train

def AE_train_fn(AE, AE_optim, loss_fn, config):
    '''
        AE training function
        Does: Initializes the training function for the AE model.
        Args: - AE (Torch neural network): Autoencoder Model
              - AE_optim (Torch optimizer function): AE optimizer (Adam or SGD)
              - loss_fn (Torch loss function): AE loss function
              - config (dict): config dictionary of parameters
    '''
    def train(x, itr, epoch):
        '''
            AE training function
            Does: Trains the AE model
            Args: x (Torch tensor): Real data input image
            Returns: list of training metrics
        '''
        # Make sure model is in training mode
        AE.train()

        # Move data to GPU (if not there already) and flatten into vector
        x = x.view(config['batch_size'], -1).to(config['gpu'])

        # Forward pass
        output = AE(x)

        # Compare output to real data
        loss = loss_fn(output, x)

        # Backprop and update weights
        AE_optim.zero_grad()
        loss.backward()
        AE_optim.step()

        # Save output periodically - concatenate the model outputs with the
        # images it was supposed to reconstruct in order to visualize the
        # model evolution during training.
        if itr % 100 == 0:
            # Arrange training data and model outputs on
            # alternating rows for easy visual comparison.
            row1 = x[0:config['sample_size']//2, :]
            row2 = output[0:config['sample_size']//2, :]
            row3 = x[config['sample_size']//2:config['sample_size'], :]
            row4 = output[config['sample_size']//2:config['sample_size'], :]
            sample = torch.cat([row1, row2, row3, row4])
            sample = sample.view(sample.size(0), 1,
                                 config['dataset'], config['dataset'])
            utils.save_sample(sample, epoch, itr, config['random_samples'])

        # Return training metrics
        metrics = { 'ae_loss' : float(loss.item()) }

        return metrics
    return train

def Conv_AE_train_fn(AE, AE_optim, loss_fn, config):
    '''
        Convolutional AE training function
        Does: Initializes the training function for the AE model.
        Args: - AE (Torch neural network): Convolutional Autoencoder Model
              - AE_optim (Torch optimizer function): AE optimizer (Adam or SGD)
              - loss_fn (Torch loss function): AE loss function
              - config (dict): config dictionary of parameters
    '''
    def train(x, itr, epoch):
        '''
            AE training function
            Does: Trains the AE model
            Args: x (Torch tensor): Real data input image
            Returns: list of training metrics
        '''
        # Make sure model is in training mode
        AE.train()

        # Move input to gpu -- no need to flatten
        x = x.to(config['gpu'])

        # Forward pass
        output = AE(x)

        # Compute L1 regularization
        # L1 = 0
        # for param in AE.parameters():
            # L1 += param.abs().sum()

        # Compare output to real data and add L1 reg.
        loss = loss_fn(output, x) # + (1e-6 * L1)

        # Backprop and update weights
        AE_optim.zero_grad()
        loss.backward()
        AE_optim.step()

        # TODO: check that the samples are saving correctly
        # Save output periodically - concatenate the model outputs with the
        # images it was supposed to reconstruct in order to visualize the
        # model evolution during training.
        if itr % 20 == 0:
            # Arrange training data and model outputs on
            # alternating rows for easy visual comparison.
            row1 = x[0:config['sample_size']//2, :]
            row2 = output[0:config['sample_size']//2, :]
            row3 = x[config['sample_size']//2:config['sample_size'], :]
            row4 = output[config['sample_size']//2:config['sample_size'], :]
            sample = torch.cat([row1, row2, row3, row4])
            sample = sample.view(sample.size(0), 1,
                                 config['dataset'], config['dataset'])
            utils.save_sample(sample, epoch, itr, config['random_samples'])

        # Return training metrics
        metrics = { 'ae_loss' : float(loss.item()) }

        return metrics
    return train
