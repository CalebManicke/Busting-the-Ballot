#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=30G

#SBATCH --mail-type=END
#SBATCH --mail-user=caleb.manicke@uconn.edu

python3 setup.py