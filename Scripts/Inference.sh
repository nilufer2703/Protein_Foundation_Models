#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 4
#SBATCH --output=example_1.out

set -euo pipefail

# source activate mlfold


folder_with_pdbs="/dapustor/nilufer/jonathan/ProteinMPNN/flip2_data/input_pdb"

output_dir="/dapustor/nilufer/jonathan/ProteinMPNN/Results/3_to_Many_otf_inactive_sequence_3_mutations_bn001"

mkdir -p "$output_dir"


path_for_parsed_chains="${output_dir}/parsed_pdbs.jsonl"


# ==========================
# Parse structures
# ==========================

python /dapustor/nilufer/ProteinMPNN/helper_scripts/parse_multiple_chains.py \
    --input_path "$folder_with_pdbs" \
    --output_path "$path_for_parsed_chains"


# ==========================
# ProteinMPNN inference
# ==========================

python /dapustor/nilufer/ProteinMPNN/protein_mpnn_run.py \
    --path_to_model_weights "/dapustor/nilufer/ProteinMPNN/training/3_to_Many_otf_inactive_sequence_3_mutations_bn001/model_weights" \
    --model_name "best" \
    --jsonl_path "$path_for_parsed_chains" \
    --out_folder "$output_dir" \
    --num_seq_per_target 5 \
    --sampling_temp "0.1" \
    --seed 37 \
    --save_prob 1 \
    --save_score 1 \
    --batch_size 1


# ==========================
# Extract log probabilities
# ==========================

python /dapustor/nilufer/jonathan/ProteinMPNN/flip2_data/codes/prob.py \
    --input_folder "${output_dir}/probs" \
    --output_folder "${output_dir}/log_probs"