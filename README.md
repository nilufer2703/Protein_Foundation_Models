# ProteinMPNN Pipeline

Main Repository:  
https://github.com/dauparas/ProteinMPNN/tree/main


# 1. Environment Setup

Activate the shared conda environment:

```bash
conda activate /dapustor/nilufer/jonathan/mlfold
```

---

# 2. ProteinMPNN Fine-tuning

## Base Model Weights

The original ProteinMPNN model weights are stored at:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/vanilla_model_weights
```

---

## Run Fine-tuning

Script:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/Scripts/finetune.sh
```

Training code:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/training/On_the_fly_real_negatives.py
```

---

## Fine-tuning Outputs

Each experiment is saved under:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/Results/{Exp_name}
```

where `{Exp_name}` corresponds to the experiment name.

This folder contains:

- trained model checkpoints
- run logs
- inference outputs
- evaluation results

---

# 3. ProteinMPNN Inference

## Run Inference

Script:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/Scripts/Inference.sh
```

ProteinMPNN inference code:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/protein_mpnn_run.py
```

---

## Inference Output

After inference, log probabilities are extracted and stored at:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/Results/{Exp_name}/log_probs
```

These files are used for downstream ranking and Top-K evaluation.

---

# 4. Training Evaluation

Used for plotting training and validation metrics.

## Run Evaluation

Script:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/Scripts/Evaluation.sh
```

---

## Input

Training log files are found inside:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/Results/{Exp_name}/run_logs
```

---

## Output

Generated plots are saved to:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/Results/{Exp_name}/train_plots
```

Generated metrics include:

- Mask_Sppo loss
- Train Accuracy 
- Validation Accuracy

---

# 5. Top-K Overlap Evaluation

Used to compare ProteinMPNN ranking performance with experimental activity ranking.

## Code

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/flip2_data/codes/Top10.py
```

---

## Input

Input files are generated after inference:

```bash
/dapustor/nilufer/jonathan/ProteinMPNN/Results/{Exp_name}/log_probs
```

These contain residue-level log probabilities predicted by ProteinMPNN.

---

## Output

Top-K analysis produces:

- Top-10 overlap graphs for Train , validation and Test


Results are saved in the output directory specified inside `Top10.py`. 

---

# General Workflow

```text
Activate environment
        |
        v
Fine-tune ProteinMPNN
        |
        v
Run inference
        |
        v
Extract log probabilities
        |
        v
Evaluate training metrics
        |
        v
Run Top-K overlap analysis
```

For a new experiment, replace:

```text
{Exp_name}
```

with the corresponding experiment folder name.

Example:

```text
3_to_Many_otf_inactive_sequence_3_mutations_bn001
```
