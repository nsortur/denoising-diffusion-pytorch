#!/bin/bash

#SBATCH --gres=gpu:volta:2
#SBATCH --partition=gaia
#SBATCH -o outputs/dermdiffusion-%j
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=256GB

python train_hydra.py
