#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--gpu 0 \
--MNIST True \
--batch_size 16 \
--num_epochs 100 \
--shuffle True \
--drop_last True \
--num_workers 8 \
--model ae \
--n_layers 4 \
--ae_lr 1e-4 \
--ae_opt adam \
--beta 0.5 \
--save_root /media/hdd1/kai/particle_generator/experiments/