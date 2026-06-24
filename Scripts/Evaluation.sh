#!/usr/bin/env bash
set -euo pipefail

PYTHON=python

SCRIPT="/dapustor/nilufer/jonathan/ProteinMPNN/flip2_data/codes/Accuracy.py"

LOG_FILE="/dapustor/nilufer/ProteinMPNN/training/3_to_Many_otf_inactive_sequence_3_mutations_bn001/run_logs/proteinmpnn_flip2_otf_inactive_sequence_3_mutations_bn001_20260623_115009.log"

OUT_DIR="/dapustor/nilufer/jonathan/ProteinMPNN/Results/3_to_Many_otf_inactive_sequence_3_mutations_bn001/train_plots"

MODEL_NAME="3_to_Many_otf_inactive_sequence_3_mutations"

echo "Running accuracy plot script..."
echo "Log file   : $LOG_FILE"
echo "Output dir : $OUT_DIR"
echo "Model name : $MODEL_NAME"

$PYTHON "$SCRIPT" \
  --log_file "$LOG_FILE" \
  --out_dir "$OUT_DIR" \
  --model_name "$MODEL_NAME"

echo "Done."