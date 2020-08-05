############################################################################
# train.py
# Author: Kai Stewart
# Tufts University - Department of Physics
# 01.08.2020
# - This script instantiates G as a multi-layer perceptron and computes
#       an optimal transport plan to fit G to the LArCV data distribution.
# - Use of a full dataloader is due to the fact that methods that form only
#       batch-to-batch transports using samples from a feed-forward
#       distribution are biased and do not exactly minimize the
#       Wasserstein distance.
############################################################################
# Implementation of Explicit Wasserstein Minimization described in:
# @article{1906.03471,
#   Author = {Yucheng Chen and Matus Telgarsky and Chao Zhang and Bolton Bailey
#             and Daniel Hsu and Jian Peng},
#   Title  = {A gradual, semi-discrete approach to generative network training
#             via explicit Wasserstein minimization},
#   Year   = {2019},
#   Eprint = {arXiv:1906.03471},
# }
############################################################################
# Algorithm 1: Optimal Transport Solver
# Input: Feed-forward distribution from G, training dataset
# Output: psi (optimal transport solver)
# This algorithm operates over the whole dataset and is O(N) complex
############################################################################
# Algorithm 2: Fitting Optimal Transport Plan
# Input: Sampling distribution, old generator function, Transfer plan
# Output: new generator function with updated parameters
############################################################################

# Sys Imports
import os
import time
import errno
import shutil
from tqdm import tqdm, trange
from datetime import datetime

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
import torch.distributions as tdist

# My stuff
from ewm import ewm_G
import utils
import argparser
import setup_model

# torch.backends.cudnn.benchmark = True
def line(length):
    return '-'*length

def train(config):
    '''
        Training function for EWM generator model.
    '''
    # Create python version of cpp operation
    # (Credit: Chen, arXiv:1906.03471, GitHub: https://github.com/chen0706/EWM)
    from torch.utils.cpp_extension import load
    my_ops = load(name = "my_ops",
                  sources = ["../W1_extension/my_ops.cpp",
                             "../W1_extension/my_ops_kernel.cu"],
                  verbose = False)
    import my_ops

    train_cpkt = True
    if train_cpkt: state = torch.load('/home/plutku01/projects/particle_generator/saved_gen.pth')


    # Set up GPU device ordinal - if this fails, use CUDA_LAUNCH_BLOCKING environment param...
    device = torch.device(config['gpu'])

    # Get model kwargs
    emw_kwargs = setup_model.ewm_kwargs(config)

    # Setup model on GPU
    G = ewm_G(**emw_kwargs).to(device) 
    if train_ckpt == True:
        G.load_state_dict(state['state_dict'])
    else: 
        G.weights_init()
    
    print(G)
    input('Press any key to launch')

    # Setup model optimizer
    model_params = {'g_params': G.parameters()}
    G_optim = utils.get_optim(config, model_params)
    if train_ckpt == True: 
        G_optim.load_state_dict(state['optimizer'])

    # Set up full_dataloader (single batch)
    dataloader = utils.get_dataloader(config) # Full Dataloader
    dset_size  = len(dataloader) # 10000
    print('dset_size:', dset_size)

    # Flatten the dataloader into a Tensor of shape [dset_size, l_dim]
    dataloader = dataloader.view(dset_size, -1).to(device)

    # Set up psi optimizer
    if train_ckpt == True:
        psi = state['psi']
    else:
        psi = torch.zeros(dset_size, requires_grad=True).to(device).detach().requires_grad_(True).to(device)
        
    psi_optim = torch.optim.Adam([psi], lr=config['psi_lr'])
    if train_ckpt == True:
        psi_optim.load_state_dict(state['psi_optim'])

    # Set up directories for saving training stats and outputs
    config = utils.directories(config)

    # Set up dict for saving checkpoints
    checkpoint_kwargs = {'G':G, 'G_optim':G_optim}
    train_checkpoint_kwargs = {'G':G, 'G_optim':G_optim,
                                'psi':psi, 'psi_optim':psi_optim}

    # Variance argument for the tessellation vectors
    tess_var = config['tess_var']**0.5
    
    # Compute the stopping criterion using set of test vectors
    # and computing the 'ideal' loss between the test/target.
    print(line(60))
    print("Computing stopping criterion")
    print(line(60))
    stop_criterion = []
    test_loader = utils.get_test_loader(config)
    for _, test_vecs in enumerate(test_loader):
        # Add Gaussian noise to test_vectors
        test_vecs = test_vecs.view(config['batch_size'], -1).to(device) # 'Perfect' generator model
        t1 = tess_var*torch.randn(test_vecs.shape[0], test_vecs.shape[1]).to(device)
        test_vecs += t1
        # Add Gaussian noise to target data
        t2 = tess_var*torch.randn(dataloader.shape[0], dataloader.shape[1]).to(device)
        test_target  = dataloader + t2
        # Compute the stop score
        stop_score = my_ops.l1_t(test_vecs, test_target)
        stop_loss = -torch.mean(stop_score)
        stop_criterion.append(stop_loss.cpu().detach().numpy())
    del test_loader
    # Set stopping criterion variables
    stop_min, stop_mean, stop_max = np.min(stop_criterion), np.mean(stop_criterion), np.max(stop_criterion)
    print(line(60))
    print('Stop Criterion: min: {}, mean: {}, max: {}'.format(round(stop_min, 3), round(stop_mean, 3), round(stop_max, 3)))
    print(line(60))

    # Set up stats logging
    hist_dict = {'hist_min':[], 'hist_max':[], 'ot_loss':[]}
    losses    = {'ot_loss': [], 'fit_loss': []}
    history   = {'dset_size': dset_size, 'epoch': 0, 'iter': 0, 'losses'   : losses, 'hist_dict': hist_dict}
    config['early_end'] = (200, 320) # Empirical stopping criterion from EWM author
    stop_counter = 0
    
    # Set up progress bar for terminal output and enumeration
    if train_ckpt == True:
        epoch_bar  = tqdm([i for i in range(state['epoch'], config['num_epochs'])])
    else:
        epoch_bar  = tqdm([i for i in range(config['num_epochs'])])

    stop = False 
    # Training Loop
    for epoch, _ in enumerate(epoch_bar):

        history['epoch'] = epoch

        # Set up memory lists: 
        #     - mu: simple feed-forward distribution 
        #     - transfer: transfer plan given by lists of indices
        # Rule-of-thumb: do not save the tensors themselves: instead, save the 
        #                data as a list and covert it to a tensor as needed.
        mu = [0] * config['mem_size']
        transfer = [0] * config['mem_size']
        mem_idx = 0

        # Compute the Optimal Transport Solver
        for ots_iter in range(0, dset_size//2): #was 1
            history['iter'] = ots_iter

            psi_optim.zero_grad()

            # Generate samples from feed-forward distribution
            z_batch = torch.randn(config['batch_size'], config['z_dim']).to(device)
            y_fake  = G(z_batch) # [B, dset_size]
            
            # Add Gaussian noise to the output of the generator function and to the data with tessellation vectors
            t1 = tess_var*torch.randn(y_fake.shape[0], y_fake.shape[1]).to(device)
            t2 = tess_var*torch.randn(dataloader.shape[0], dataloader.shape[1]).to(device)
            
            y_fake  += t1
            dataloader += t2
            
            # Compute the W1 distance between the model output and the target distribution
            score = my_ops.l1_t(y_fake, dataloader) - psi
            phi, hit = torch.max(score, 1)

            # Remove the tesselation from the dataloader
            dataloader -= t2
            
            # Standard loss computation
            # This loss defines the sample mean of the marginal distribution
            # of the dataset. This is the only computation that generalizes.
            loss = -torch.mean(psi[hit])

            # Backprop
            loss.backward()
            psi_optim.step()

            # Update memory tensors
            mu[mem_idx] = z_batch.data.cpu().numpy().tolist()
            #print("OTS:", mem_idx, np.asarray(mu[mem_idx]).shape)
            transfer[mem_idx] = hit.data.cpu().numpy().tolist()
            mem_idx = (mem_idx + 1) % config['mem_size'] 

            # Update losses
            history['losses']['ot_loss'].append(loss.item())

            if (ots_iter % 500 == 0):
                avg_loss = np.mean(history['losses']['ot_loss'])
                print('OTS Iteration {} | Epoch {} | Avg Loss Value: {}'.format(ots_iter, epoch, round(avg_loss, 3)))
#             if (iter % 2000 == 0):
#                 # Display histogram stats
#                 hist_dict, stop = utils.update_histogram(transfer, history, config)
#                 # Emperical stopping criterion
#                 if stop:
#                     break

            if ots_iter > (dset_size//3):
                if  stop_min <= np.mean(history['losses']['ot_loss']) <= stop_max:
                    stop_counter += 1
                    print("stopped")
                    #stop = True
                    #train_state = utils.get_checkpoint(history['epoch'], checkpoint_kwargs, config)
                    #torch.save(train_state, '/home/plutku01/projects/particle_generator/saved_gen.pth')
                    #break
        
        '''
        if stop == True:
            break
        '''
        # Compute the Optimal Fitting Transport Plan
        for fit_iter in range(config['mem_size']):# - 1):
            G_optim.zero_grad()
            # print("fit_iter: ", fit_iter)
            # print("mu[fit_iter]: ", type(mu[fit_iter]))
            #inf = np.asarray(mu[fit_iter])
            #print("FIT:", fit_iter, inf.shape)

            # Retrieve stored batch of generated samples
            z_batch = torch.tensor(mu[fit_iter]).to(device)
            #print(z_batch.shape)
            y_fake  = G(z_batch) # G'(z)
            
            # Get Transfer plan from OTS: T(G_{t-1}(z))
            t_plan = torch.tensor(transfer[fit_iter]).to(device)
            y0_hit = dataloader[t_plan].to(device)
            
#            Tesselate the output of the generator function and the data
#             t1 = tess_var*torch.randn(y_fake.shape[0], y_fake.shape[1]).to(device)
#             t2 = tess_var*torch.randn(y0_hit.shape[0], y0_hit.shape[1]).to(device)
            
#             y_fake *= t1
#             y0_hit *= t1
            
            # Compute Wasserstein distance between G and T
            G_loss = torch.mean(torch.abs(y0_hit - y_fake)) * config['l_dim']

            # Backprop
            G_loss.backward() # Gradient descent
            G_optim.step()

            # Update losses
            history['losses']['fit_loss'].append(G_loss.item())

            # Check if best loss value and save checkpoint
            if 'best_loss' not in history:
                history.update({ 'best_loss' : G_loss.item() })

            best = G_loss.item() < (history['best_loss'] * 0.70)
            if best:
                history['best_loss'] = G_loss.item()
                checkpoint = utils.get_checkpoint(history['epoch'], checkpoint_kwargs, config)
                utils.save_checkpoint(checkpoint, config)

            if (fit_iter % 500 == 0):
                avg_loss = np.mean(history['losses']['fit_loss'])
                print('FIT Iteration {} | Epoch {} | Avg Loss Value: {}'.format(fit_iter, epoch, round(avg_loss,3)))

    # Save a checkpoint at end of training
    checkpoint = utils.get_checkpoint(history['epoch'], checkpoint_kwargs, config)
    utils.save_checkpoint(checkpoint, config)

    # Save training data to csv's after training end
    utils.save_train_hist(history, config, times=None, histogram=history['hist_dict'])
    print("Stop Counter Triggered {} Times".format(stop_counter))

    # For Aiur
    print("I see you have an appetite for destruction.")
    print("And you have learned to use your illusion.")
    print("But I find your lack of control disturbing.")
