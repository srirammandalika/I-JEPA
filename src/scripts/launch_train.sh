#!/usr/bin/env bash
# scripts/launch_train.sh
# Wrapper to launch the full distilled I-JEPA training pipeline:
#   1) Student distillation
#   2) Contextual planner
#   3) Latent dynamics & value net

set -euo pipefail

# --- 1) Project root (assumes this script lives in scripts/) ---
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- 2) User arguments (with defaults) ---
TRAIN_ROOT=${1:-"/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/train/"}
VAL_ROOT=${2:-"/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/train/"}

# --- 3) Hyperparameters ---
BATCH_SIZE=64
STUDENT_EPOCHS=10
PLANNER_EPOCHS=15
DYNAMICS_EPOCHS=15
# --- 3) Optional hyperparameters (overrides) ---
# Uncomment to override defaults
# BATCH_SIZE=64
# STUDENT_EPOCHS=100
# PLANNER_EPOCHS=50
# DYNAMICS_EPOCHS=50
# Uncomment to override defaults

# --- 4) Device (optional override via env DEVICE) ---
DEVICE=${DEVICE:-}

echo "Project root: $PROJECT_ROOT"
echo "Train data:   $TRAIN_ROOT"
echo "Val data:     $VAL_ROOT"
echo "Device:       ${DEVICE:-auto}"
echo

# --- 5) Train Student ---
echo "===== Training Student (System 1) ====="
python -m src.training.train_student \
  --train_root "$TRAIN_ROOT" \
  --val_root   "$VAL_ROOT"   \
  --batch_size $BATCH_SIZE   \
  --epochs     $STUDENT_EPOCHS \
  ${DEVICE:+--device $DEVICE}
echo

# --- 6) Train Contextual Planner ---
echo "===== Training Planner (System 2) ====="
python -m src.training.train_planner \
  --train_root "$TRAIN_ROOT" \
  --val_root   "$VAL_ROOT"   \
  --batch_size $BATCH_SIZE   \
  --epochs     $PLANNER_EPOCHS \
  ${DEVICE:+--device $DEVICE}
echo

# --- 7) Train Latent Dynamics & Value Net ---
echo "===== Training Dynamics & Value Net ====="
python -m src.training.train_dynamics \
  --train_root "$TRAIN_ROOT" \
  --batch_size $BATCH_SIZE   \
  --epochs     $DYNAMICS_EPOCHS \
  ${DEVICE:+--device $DEVICE}
echo

echo "All training stages completed successfully."
