#!/bin/bash -l
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=35
#SBATCH -t 10:00:00
#SBATCH --job-name=ecg_cnn_mix4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shaofeng.wang@ucdconnect.ie
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

# Data root on HPC (you will upload shards here)
export FYP_DATA_DIR="$HOME/scratch/data"

# Example paths (adjust to your real locations)
PAIRS_DIR="$SLURM_SUBMIT_DIR/splits/splits"

python -u train_cls_baseline.py \
  --tasks mitdb_beat incartdb_beat qtdb_beat sddb_beat \
  --pairs-train "$PAIRS_DIR/mix4_beat_fixedsplit_pairs_train.npy" \
  --pairs-val   "$PAIRS_DIR/mix4_beat_fixedsplit_pairs_val.npy" \
  --pairs-test  "$PAIRS_DIR/mix4_beat_fixedsplit_pairs_test.npy" \
  --data-root   "$FYP_DATA_DIR/downstream_shards_with_groups" \
  --out         "./outputs" \
  --run-name    "$(date +%F_%H%M)_mix4_cnn_baseline" \
  --model       cnn_small \
  --num-classes 5 \
  --input-len   300 \
  --epochs      50 \
  --batch-size  512 \
  --lr          1e-3 \
  --weight-decay 1e-4 \
  --num-workers 8 \
  --seed        42 \
  --max-steps   0
