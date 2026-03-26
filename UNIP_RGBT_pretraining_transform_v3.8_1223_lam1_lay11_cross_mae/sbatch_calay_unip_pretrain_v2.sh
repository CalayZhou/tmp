#!/bin/bash

#SBATCH --job-name=mae-l_distill_unip-s_calay
#SBATCH --time=48:00:00
#SBATCH --gpus 2 -C gpu_32g
#SBATCH --output=output_unip_calay.log
#SBATCH --error=error_unip_calay.log

# 修复conda环境初始化
source ~/.bashrc                   # 加载bash配置，其中应该包含conda初始化
module load Miniforge3

# 正确的conda激活方式
eval "$(conda shell.bash hook)"    # 关键：初始化conda的shell环境
conda activate unip_pre

# 调试信息：检查环境
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
echo "Conda环境: $CONDA_DEFAULT_ENV"

# 检查关键包是否可导入
#ython -c "import mmengine; print('mmengine版本:', mmengine.__version__)"
#ython -c "import mmseg; print('mmseg版本:', mmseg.__version__)"

# 运行训练脚本
sh train_scripts/mae-l_distill_unip-s_calay.sh
#python tools/train.py configs/mae/mae-small_upernet_8xb2-amp-320k_ade20k-768x768_classbalance_unip.py
#bash tools/dist_train.sh  configs/mae/mae-small_upernet_8xb2-amp-320k_ade20k-768x768_classbalance_unip.py 2
