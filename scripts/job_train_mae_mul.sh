#!/bin/bash -l
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=35
#SBATCH -t 10:00:00
#SBATCH --job-name=ecg_mae_multidb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shaofeng.wang@ucd.ie
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs

echo "HOST=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION  GRES=$SLURM_JOB_GRES"
nvidia-smi || true
python - << 'EOF'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
EOF

# source ~/.conda/envs/fyp/bin/activate   # 改成你自己的 venv

# 数据根目录
export FYP_DATA_DIR=$HOME/projects/FYP

# 重点改动：--npy -> --data（给目录 or glob or 多个文件）
python -u train_mae_v2.py \
  --data ./data/pretrain/pretrain_singlelead_500hz_10s \
  --recursive \
  --out ./outputs \
  --run-name $(date +%F_%H%M)_mae_multidb \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --num-workers 2 \
  --mask-ratio 0.6 \
  --patch-size 10 \
  --d-model 128 \
  --depth 4 \
  --n-heads 4 \
  --dim-ff 256 \
  --dropout 0.1 \
  --per-file-limit 0 \
  --global-limit 0
