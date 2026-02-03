#!/bin/bash -l
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=35
#SBATCH -t 23:59:59
#SBATCH --job-name=ecg_nonfm_mix
#SBATCH --array=0-1%1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shaofeng.wang@ucdconnect.ie
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# ############### How to submit ###############
# Step 1: Load anaconda3
#     module load anaconda3
#
# Step 2: Activate FYP env
#     conda activate fyp
#
# Step 3: Submit job to HPC
#     cd path/to/FYP
#     sbatch --partition=dev ./scripts/your_script_name.sh
#     sbatch --partition=gpu ./scripts/your_script_name.sh
#
# Step 4: Check job state
#     sacct -j jobid
#
# Step 5: Cancel job
#     scancel jobid
# ############### How to submit ###############

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

echo "HOST=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION  GRES=$SLURM_JOB_GRES"
echo "JOBID=$SLURM_JOB_ID  ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
nvidia-smi || true

python - << 'EOF'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
EOF

# ------------------------------
# Data paths on HPC
# ------------------------------
export FYP_DATA_DIR="$HOME/scratch"
PAIRS_DIR="$HOME/scratch/splits/splits"

# ------------------------------
# Choose model by array id
# ------------------------------
MODEL="cnn_small"
MODEL_TAG="cnn"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL="cnn_small"
    MODEL_TAG="cnn"
    ;;
  1)
    MODEL="lstm_cnn_small"
    MODEL_TAG="lstm_cnn"
    ;;
  *)
    echo "[ERROR] Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

# ------------------------------
# Full training config (same as yours)
# ------------------------------
EPOCHS=100
BATCH_SIZE=512
LR=1e-3
WD=1e-4
NUM_WORKERS=8
SEED=42
MAX_STEPS=0
NUM_CLASSES=5
INPUT_LEN=300

TS="$(date +%F_%H%M%S)"
RUN_NAME="${TS}_mix9_${MODEL_TAG}_baseline_nonfm_E${EPOCHS}_seed${SEED}"

echo "[INFO] MODEL=${MODEL}"
echo "[INFO] RUN_NAME=${RUN_NAME}"

python -u train_cls_baseline.py \
  --tasks mitdb_beat incartdb_beat ltafdb_beat qtdb_beat ltdb_beat nstdb_beat sddb_beat stdb_beat svdb_beat \
  --pairs-train "${PAIRS_DIR}/mix_beat_fixedsplit_pairs_train.npy" \
  --pairs-val   "${PAIRS_DIR}/mix_beat_fixedsplit_pairs_val.npy" \
  --pairs-test  "${PAIRS_DIR}/mix_beat_fixedsplit_pairs_test.npy" \
  --data-root   "${FYP_DATA_DIR}/downstream_shards_with_groups" \
  --out         "./outputs" \
  --run-name    "${RUN_NAME}" \
  --model       "${MODEL}" \
  --num-classes "${NUM_CLASSES}" \
  --input-len   "${INPUT_LEN}" \
  --epochs      "${EPOCHS}" \
  --batch-size  "${BATCH_SIZE}" \
  --lr          "${LR}" \
  --weight-decay "${WD}" \
  --num-workers "${NUM_WORKERS}" \
  --seed        "${SEED}" \
  --max-steps   "${MAX_STEPS}"

echo "[INFO] Done: ${RUN_NAME}"
