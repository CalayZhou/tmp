#!/bin/bash

#SBATCH --job-name=mae-l_distill_unip-s_calay_copy
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:6000ada:4
#SBATCH --output=output_unip_calay_%j.log  # 添加作业ID避免覆盖
#SBATCH --error=error_unip_calay_%j.log    # 添加作业ID避免覆盖
#SBATCH --cpus-per-task=16             # 每个任务的CPU核心数
#SBATCH --mem=128G                     # 内存请求


# 设置工作目录（建议添加）
cd $SLURM_SUBMIT_DIR

echo "开始时间: $(date)"
echo "工作目录: $SLURM_SUBMIT_DIR"
echo "作业ID: $SLURM_JOB_ID"

# 环境初始化
source ~/.bashrc
module load Miniforge3

# conda激活
eval "$(conda shell.bash hook)"
if conda activate unip_pre; then
    echo "成功激活conda环境: unip_pre"
else
    echo "错误: 无法激活conda环境 unip_pre"
    exit 1
fi

# 调试信息
echo "=== 环境检查 ==="
echo "主机名: $(hostname)"
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
echo "Conda环境: $CONDA_DEFAULT_ENV"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 检查关键依赖（取消注释并修正）
echo "=== 依赖检查 ==="
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"

# 运行训练脚本
echo "=== 开始训练 ==="
if sh train_scripts/mae-l_distill_unip-s_rgbt_server_calay_slurm_quick_resume.sh; then
    echo "训练成功完成"
else
    echo "训练失败，退出码: $?"
    exit 1
fi

echo "结束时间: $(date)"
