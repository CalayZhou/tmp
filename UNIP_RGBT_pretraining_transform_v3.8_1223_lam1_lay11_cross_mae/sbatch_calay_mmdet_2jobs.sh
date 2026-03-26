#!/bin/bash
#SBATCH --job-name=vitdet_mae
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=output.log
#SBATCH --error=error.log

# ====== 环境 ======
source ~/.bashrc
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate mmdet

echo "Job started on node $SLURM_NODEID"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# ====== 并行启动两个任务 ======

# Task 1 -> GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 \
bash tools/dist_train.sh \
    projects/ViTDet/configs/vitdet_mask-rcnn_vit-s-mae_lsj-100e_lrreduce.py 2 &

# Task 2 -> GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 \
bash tools/dist_train2.sh \
    projects/ViTDet/configs/vitdet_mask-rcnn_vit-s-mae_lsj-100e_lrreduce2.py 2 &

# 等待两个训练完成
wait
echo "Both tasks completed!"

