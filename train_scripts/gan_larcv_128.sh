#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--gpu 0 \
--dataset 128 \
--batch_size 50 \
--num_epochs 100 \
--sample_size 16 \
--shuffle True \
--drop_last True \
--num_workers 8 \
--model gan \
--n_hidden 512 \
--n_layers 4 \
--g_lr 1e-4 \
--g_opt adam \
--z_dim 100 \
--d_lr 1e-4 \
--d_opt adam \
--beta 0.5 \
--data_root /media/disk1/kai/larcv_png_data/ \
--save_root /media/hdd1/kai/particle_generator/experiments/
