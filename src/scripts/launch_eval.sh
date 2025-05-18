#!/usr/bin/env bash
# scripts/launch_eval.sh
# Wrapper to run inference/evaluation for the distilled I-JEPA pipeline:
#   - System 1 (student) fast path
#   - System 2 (deep planner) fallback
#   Collects outputs and optionally computes metrics.

set -euo pipefail

# --- 1) Project root ---
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- 2) User arguments / defaults ---
DATA_ROOT=${1:-"/path/to/cifar10/val"}
OUTPUT_DIR=${2:-"$PROJECT_ROOT/evaluation"}
DEVICE=${DEVICE:-}

# --- 3) Checkpoint locations ---
STUDENT_CKPT="checkpoints/student.pth"
TEACHER_CKPT="checkpoints/teacher_enc.pth"
PLANNER_CKPT="checkpoints/planner.pth"
DYNAMICS_CKPT="checkpoints/dynamics.pth"
VALUE_CKPT="checkpoints/value_net.pth"

# --- 4) Create output directory ---
mkdir -p "$OUTPUT_DIR"

echo "Project root:   $PROJECT_ROOT"
echo "Data for eval:  $DATA_ROOT"
echo "Output dir:     $OUTPUT_DIR"
echo "Device:         ${DEVICE:-auto}"
echo

# --- 5) Run inference ---
echo "===== Running Inference Pipeline ====="
python -m src.inference.run_inference \
  --data_root "$DATA_ROOT" \
  --student_ckpt "$STUDENT_CKPT" \
  --teacher_ckpt "$TEACHER_CKPT" \
  --planner_ckpt "$PLANNER_CKPT" \
  --dynamics_ckpt "$DYNAMICS_CKPT" \
  --value_ckpt "$VALUE_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  ${DEVICE:+--device $DEVICE}

echo
echo "Inference complete. Outputs saved under $OUTPUT_DIR."
