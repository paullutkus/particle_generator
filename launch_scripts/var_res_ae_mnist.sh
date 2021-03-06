#!/bin/bash
# Def: ldim 102 batch_size 250 n layers 3
#--data_root /home/plutku01/data/cv/single_particle/train/ \
CUDA_VISIBLE_DEVICES=0,1 python3 ../train.py \
--MNIST True \
--gpu 0 \
--dataset 64 \
--batch_size 250 \
--num_epochs 150 \
--sample_size 16 \
--shuffle True \
--drop_last True \
--num_workers 8 \
--model var_res_ae \
--depth 32 \
--n_layers 3 \
--l_dim 384 \
--ae_lr 1e-3 \
--ae_opt adam \
--loss_fn bce \
--beta 0.5 \
--save_root /home/plutku01/projects/particle_generator/experiments/
