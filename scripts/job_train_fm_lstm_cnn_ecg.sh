#!/bin/bash -l
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 47:59:59
#SBATCH --job-name=ecg_ablate_ls_full
#SBATCH --array=0-5
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

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID}  ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true

# =============================
# USER CONFIG (EDIT THESE)
# =============================

export FYP_DATA_DIR="$HOME/scratch"
PAIRS_DIR="$HOME/scratch/splits/splits"
FM_CKPT="$HOME/projects/FYP/outputs/runs/2025-12-28_1610_mae_multidb/checkpoints/best.pt"

PY_ENTRY="python -u train_cls_baseline.py"

TASKS=(
  mitdb_beat
  incartdb_beat
  ltafdb_beat
  qtdb_beat
  ltdb_beat
  nstdb_beat
  sddb_beat
  stdb_beat
  svdb_beat
)

# =============================
# FULL TRAIN BUDGET (CHANGE IF NEEDED)
# =============================
EPOCHS=100          # <-- full training epochs (e.g., 50 or 100)
MAX_STEPS=0         # <-- MUST be 0 for full training
BATCH_SIZE=512
NUM_WORKERS=8
SEED=42

LR=1e-3
WD=1e-4
NUM_CLASSES=5
INPUT_LEN=300

# shared imbalance params
ALPHA=0.5
FOCAL_GAMMA=2.0

# =============================
# MAP array id -> (sampling, loss)
# =============================
SAMPLING="none"
LOSS="ce"
SAMPLING_ARGS=()
LOSS_ARGS=()

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # none + ce
    SAMPLING="none"; LOSS="ce"
    ;;
  1)
    # wrs + ce
    SAMPLING="wrs"; LOSS="ce"
    SAMPLING_ARGS=(--sampling wrs --sampling-replacement --weights-alpha "${ALPHA}")
    ;;
  2)
    # none + wce
    SAMPLING="none"; LOSS="wce"
    LOSS_ARGS=(--weights-alpha "${ALPHA}")
    ;;
  3)
    # wrs + wce
    SAMPLING="wrs"; LOSS="wce"
    SAMPLING_ARGS=(--sampling wrs --sampling-replacement --weights-alpha "${ALPHA}")
    LOSS_ARGS=(--weights-alpha "${ALPHA}")
    ;;
  4)
    # none + focal
    SAMPLING="none"; LOSS="focal"
    LOSS_ARGS=(--weights-alpha "${ALPHA}" --focal-gamma "${FOCAL_GAMMA}")
    ;;
  5)
    # wrs + focal
    SAMPLING="wrs"; LOSS="focal"
    SAMPLING_ARGS=(--sampling wrs --sampling-replacement --weights-alpha "${ALPHA}")
    LOSS_ARGS=(--weights-alpha "${ALPHA}" --focal-gamma "${FOCAL_GAMMA}")
    ;;
  *)
    echo "[ERROR] Unknown array task id: ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

# =============================
# Unique run name
# =============================
TS="$(date +%F_%H%M%S)"
RUN_NAME="${TS}_mix9_fm_lstm_cnn_${SAMPLING}_${LOSS}_E${EPOCHS}_seed${SEED}"

echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] SAMPLING=${SAMPLING}  LOSS=${LOSS}"
echo "[INFO] EPOCHS=${EPOCHS}  MAX_STEPS=${MAX_STEPS} (full)"
echo "[INFO] DATA_ROOT=${FYP_DATA_DIR}/downstream_shards_with_groups"
echo "[INFO] PAIRS_DIR=${PAIRS_DIR}"
echo "[INFO] FM_CKPT=${FM_CKPT}"

TASK_ARGS=("${TASKS[@]}")

# =============================
# Run FULL training
# =============================
${PY_ENTRY} \
  --tasks "${TASK_ARGS[@]}" \
  --pairs-train "${PAIRS_DIR}/mix_beat_fixedsplit_pairs_train.npy" \
  --pairs-val   "${PAIRS_DIR}/mix_beat_fixedsplit_pairs_val.npy" \
  --pairs-test  "${PAIRS_DIR}/mix_beat_fixedsplit_pairs_test.npy" \
  --data-root   "${FYP_DATA_DIR}/downstream_shards_with_groups" \
  --out         "./outputs" \
  --run-name    "${RUN_NAME}" \
  --model       fm_lstm_cnn \
  --fm-ckpt     "${FM_CKPT}" \
  --fm-unfreeze-last-n 0 \
  --num-classes "${NUM_CLASSES}" \
  --input-len   "${INPUT_LEN}" \
  --epochs      "${EPOCHS}" \
  --batch-size  "${BATCH_SIZE}" \
  --lr          "${LR}" \
  --weight-decay "${WD}" \
  --num-workers "${NUM_WORKERS}" \
  --seed        "${SEED}" \
  --max-steps   "${MAX_STEPS}" \
  --loss        "${LOSS}" \
  "${SAMPLING_ARGS[@]}" \
  "${LOSS_ARGS[@]}"

echo "[INFO] Done: ${RUN_NAME}"
