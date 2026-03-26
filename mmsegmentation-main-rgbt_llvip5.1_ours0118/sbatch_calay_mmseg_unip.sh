#!/bin/bash

#SBATCH --job-name=mae-small_upernet_8xb2-amp-320k_ade20k-768x768_classbalance_unip
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:6000ada:2
#SBATCH --output output_mmseg_calay.log
#SBATCH --error error_mmseg_calay.log


# NOTE:
# For a list of helpful flags to specify above, check out Slurm Overview in slurm.md.

module load Miniforge3  
conda init
conda activate mmseg
python tools/train.py configs/mae/mae-small_upernet_8xb2-amp-320k_ade20k-768x768_classbalance_unip.py
#bash tools/dist_train.sh  configs/mae/mae-large_upernet_8xb2-amp-320k_ade20k-768x768.py  2 
