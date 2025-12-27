#!/bin/bash -l
#SBATCH -N 1
# SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=35
#SBATCH -t 10:00:00
#SBATCH --job-name=ecg_mae_mitdb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shaofeng.wang@ucd.ie
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

echo "HOST=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION  GRES=$SLURM_JOB_GRES"
nvidia-smi || true
python - << 'EOF'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
EOF

# 如果你用的是 conda / venv，这里要激活环境（记得改成你自己的路径）
# source ~/.conda/envs/fyp/bin/activate

# 数据根目录：建议指到项目里的 data 目录
# 之后 --npy 用相对这个目录的路径
export FYP_DATA_DIR="$HOME/scratch/data"

# submit: sbatch --partition=dev ./job_train_mae.sh
# check state: sacct -j jobid
# cancel: scancel jobid

# 这里调用的是新的入口脚本 train_mae.py
python -u ./train_mae.py \
  --npy mitdb_singlelead_500hz_10s.npy \
  --out ./outputs \
  --run-name "$(date +%F_%H%M)_mae_mitdb" \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3 \
  --num-workers 4 \
  --mask-ratio 0.6 \
  --patch-size 10 \
  --d-model 128 \
  --depth 4 \
  --n-heads 4 \
  --dim-ff 256 \
  --dropout 0.1 \
  --max-steps 50
