!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python
SCRIPT="/dapustor/nilufer/jonathan/ProteinMPNN/training/On_the_fly_real_negatives.py"

BASE_OUT_DIR="/dapustor/nilufer/jonathan/ProteinMPNN/Results/3_to_Many"

CSV_PATH_TRAIN="/dapustor/nilufer/ProteinMPNN/flip2_data/imine_reductase_three_to_many.csv"

POSITIVE_POOL_CSV="/dapustor/nilufer/ProteinMPNN/flip2_data/imine_reductase_three_to_many.csv"
REAL_NEGATIVE_CSV="/dapustor/nilufer/ProteinMPNN/flip2_data/inactive_seq_three_mutations.csv"

POSITIVE_THRESHOLD="0.5"
REAL_PAIRS_PER_STEP=3

EXP_NAME="otf_inactive_sequence_3_mutations"

BACKBONE_NOISES=(
  "0.01"
)

for bn in "${BACKBONE_NOISES[@]}"; do

  if [[ "$bn" == "0.01" ]]; then
    BN_TAG="bn001"
  else
    BN_TAG="bn000"
  fi

  REAL_PAIR_NAME="${EXP_NAME}_${BN_TAG}"

  OUT_DIR="${BASE_OUT_DIR}_${REAL_PAIR_NAME}"
  LOG_DIR="${OUT_DIR}/run_logs"
  mkdir -p "$LOG_DIR"

  RUN_TAG="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/proteinmpnn_flip2_${REAL_PAIR_NAME}_${RUN_TAG}.log"

  echo "=========================================="
  echo "Running experiment: ${REAL_PAIR_NAME}"
  echo "Backbone noise: ${bn}"
  echo "Positive pool CSV: ${POSITIVE_POOL_CSV}"
  echo "Real negative CSV: ${REAL_NEGATIVE_CSV}"
  echo "Positive threshold: ${POSITIVE_THRESHOLD}"
  echo "Real pairs per step: ${REAL_PAIRS_PER_STEP}"
  echo "Output dir: ${OUT_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "=========================================="

  "$PYTHON_BIN" "$SCRIPT" \
    --path_for_outputs "$OUT_DIR" \
    --csv_path_train "$CSV_PATH_TRAIN" \
    --train_pos_seq_coord_path /dapustor/nilufer/ProteinMPNN/flip2_data/structure \
    --flip2_pdb_id 5OCM_B \
    --ce_csv_path_train /dapustor/nilufer/YYH/ProteinMPNN/data/train_protein_mpnn_final.csv \
    --ce_pos_seq_coord_path /dapustor/nilufer/YYH/ProteinMPNN/data/train_pos_seq_coords_with_mask \
    --num_epochs 50 \
    --flip2_minibatch_size 15 \
    --max_pairs_per_step 12 \
    --ce_batch_size 1 \
    --pair_threshold 1 \
    --beta 0.2 \
    --lambda_ce 0.2 \
    --backbone_noise "$bn" \
    --flip2_backbone_noise 0 \
    --save_model_every_n_epochs 1 \
    --onthefly_pairs_per_step 12 \
    --real_pairs_per_step "$REAL_PAIRS_PER_STEP" \
    --positive_pool_csv "$POSITIVE_POOL_CSV" \
    --real_negative_csv "$REAL_NEGATIVE_CSV" \
    --positive_threshold "$POSITIVE_THRESHOLD" \
    2>&1 | tee "$LOG_FILE"

done