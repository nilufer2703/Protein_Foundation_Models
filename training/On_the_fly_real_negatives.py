import argparse
import os
import math
import pandas as pd
import copy
import time
import logging
from tqdm import tqdm   
from datetime import datetime
import glob 
import random
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import csv


from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
from model_utils_ori import featurize, loss_smoothed,loss_nll, get_std_opt, ProteinMPNN
from transformers import get_linear_schedule_with_warmup
from model_utils import NegativeSequencePool, PositiveSequencePool, build_random_real_pairs
     



def make_batch_from_pdb(pdb_id, visible_chains=[], masked_chains=[], seq_override=None, data_path=None):
    """
    Returns a list of dict.
    """
    batch_dict = {}
   
    if "_" in pdb_id:
        pdb_base, chain = pdb_id.split("_")
    else:
        pdb_base = pdb_id
        chain = None

    masked_chains = [chain]
    

    batch_dict['num_of_chains'] = 1
    batch_dict['visible_list'] = visible_chains
    batch_dict['masked_list'] = masked_chains

    # Load chain-specific file
    chain_file = os.path.join(data_path, f"{pdb_base}_{chain}.pt")
    chain_data = torch.load(chain_file)

    # Sequence handling
    seq = chain_data['seq']
    if seq_override is not None:
        seq = seq_override

    # Extract backbone atoms
    xyz = chain_data['xyz']
   
    coords_chain = {
        f'N_chain_{chain}': xyz[:,0,:],
        f'CA_chain_{chain}': xyz[:,1,:],
        f'C_chain_{chain}': xyz[:,2,:],
        f'O_chain_{chain}': xyz[:,3,:]
    }
    batch_dict[f'coords_chain_{chain}'] = coords_chain
    batch_dict[f'seq_chain_{chain}'] = seq
    batch_dict["seq"] = seq
    batch_dict["pdb_id"] = pdb_base

    return [batch_dict]


def cross_entropy_loss(S, log_probs, mask):
    
    B, L, V = log_probs.size()

    loss_per_residue = F.nll_loss(
        log_probs.view(-1, V),      # predictions
        S.view(-1),                 # target
        reduction='none'
    ).view(B, L)

    # Mean loss over all residues
    loss = torch.sum(loss_per_residue*mask, dim=1)/(torch.sum(mask, dim=1)+1e-8)
    loss = loss.mean()
    
    return loss



def mask_sppo_loss(logp_pos, logp_neg, logp_ref_pos, logp_ref_neg, S_pos, S_neg, mask, beta):
    """
    Masked SPPO loss — focuses only on differing residues between positive and negative sequences.
    """
    diff_mask_ = (S_pos != S_neg).float()  # [B, L]
    diff_mask = diff_mask_*mask
    mask_sum = diff_mask.sum(dim=1)  # [B, 1]
    mask_sum = torch.clamp(mask_sum, min=1.0)

    logp_pos_masked = (logp_pos * diff_mask).sum(dim=1)/ mask_sum
    logp_neg_masked = (logp_neg * diff_mask).sum(dim=1)/ mask_sum
    logp_ref_pos_masked = (logp_ref_pos * diff_mask).sum(dim=1)/ mask_sum
    logp_ref_neg_masked = (logp_ref_neg * diff_mask).sum(dim=1)/ mask_sum 
    

    a = beta * (logp_pos_masked - logp_ref_pos_masked)
    b = beta * (logp_neg_masked - logp_ref_neg_masked)

    loss = (a - 0.5) ** 2 + (b + 0.5) ** 2

    return loss.mean()


def preference_sppo_loss(
    logp_pos,
    logp_neg,
    logp_ref_pos,
    logp_ref_neg,
    S_pos,
    S_neg,
    mask,
    beta,
):
    return mask_sppo_loss(
        logp_pos,
        logp_neg,
        logp_ref_pos,
        logp_ref_neg,
        S_pos,
        S_neg,
        mask,
        beta,
    )


def ce_nll_loss(
    S,
    logp_all,
    mask,
):
    return cross_entropy_loss(S, logp_all, mask)

def add_backbone_noise_to_X(X, noise_std):
    if noise_std is None or noise_std <= 0:
        return X

    noise = torch.randn_like(X) * noise_std
    return X + noise

def model_log_prob_per_res(model, X, S, mask, chain_M, residue_idx, chain_encoding_all):
    logp_probs_all = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)  # [B, L, 21]
    logp_seq = logp_probs_all.gather(-1, S.unsqueeze(-1)).squeeze(-1)  # [B, L]
    return logp_seq ,logp_probs_all

def parse_boolish(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    return value in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }

def setup_logger(logfile):
    """Setup logging to both file and console"""
    logger = logging.getLogger("SPPO_ProteinMPNN")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(logfile)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def build_proteinmpnn_preference_pairs_from_batch(
    batch_rows,
    threshold,
    max_pairs=None,
):
    if len(batch_rows) < 2:
        return []

    targets = np.array(
        [float(row["target"]) for row in batch_rows],
        dtype=np.float32,
    )

    diff_matrix = np.abs(targets[:, None] - targets[None, :])
    upper_tri = np.triu(
        np.ones((len(batch_rows), len(batch_rows)), dtype=bool),
        k=1,
    )

    valid_pair_mask = (diff_matrix > threshold) & upper_tri
    valid_pair_indices = np.argwhere(valid_pair_mask)

    if len(valid_pair_indices) == 0:
        return []

    valid_pair_indices = valid_pair_indices.tolist()
    random.shuffle(valid_pair_indices)

    if max_pairs is None:
        max_pairs = len(batch_rows)

    selected_pairs = valid_pair_indices[:max_pairs]

    paired_items = []

    for i, j in selected_pairs:
        row_i = batch_rows[i]
        row_j = batch_rows[j]

        if float(row_i["target"]) > float(row_j["target"]):
            pos_row = row_i
            neg_row = row_j
        else:
            pos_row = row_j
            neg_row = row_i

        paired_items.append(
            {
                "pair_id": f"{pos_row['sequence_id']}__vs__{neg_row['sequence_id']}",
                "pdb_id": pos_row["pdb_id"],
                "pos_sequence": pos_row["sequence"],
                "neg_sequence": neg_row["sequence"],
                "pos_target": float(pos_row["target"]),
                "neg_target": float(neg_row["target"]),
                "target_gap": abs(float(pos_row["target"]) - float(neg_row["target"])),
                "pos_sequence_id": pos_row["sequence_id"],
                "neg_sequence_id": neg_row["sequence_id"],
                "neg_source": "on_the_fly",
            }
        )

    return paired_items

def safe_pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def masked_mean_score(logp, mask):
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    return ((logp * mask).sum(dim=1) / denom).mean()


def compute_pair_metrics(logp_pos, logp_neg, logp_ref_pos, logp_ref_neg, S_pos, S_neg, mask_for_loss):
    full_mask = mask_for_loss
    critical_mask = ((S_pos != S_neg).float() * mask_for_loss)

    policy_full_pos = masked_mean_score(logp_pos, full_mask)
    policy_full_neg = masked_mean_score(logp_neg, full_mask)
    ref_full_pos = masked_mean_score(logp_ref_pos, full_mask)
    ref_full_neg = masked_mean_score(logp_ref_neg, full_mask)

    policy_full_margin = policy_full_pos - policy_full_neg
    ref_full_margin = ref_full_pos - ref_full_neg

    if critical_mask.sum().item() > 0:
        policy_critical_pos = masked_mean_score(logp_pos, critical_mask)
        policy_critical_neg = masked_mean_score(logp_neg, critical_mask)
        ref_critical_pos = masked_mean_score(logp_ref_pos, critical_mask)
        ref_critical_neg = masked_mean_score(logp_ref_neg, critical_mask)

        policy_critical_margin = policy_critical_pos - policy_critical_neg
        ref_critical_margin = ref_critical_pos - ref_critical_neg
    else:
        policy_critical_margin = torch.zeros((), device=logp_pos.device)
        ref_critical_margin = torch.zeros((), device=logp_pos.device)

    return {
        "policy_full_margin": float(policy_full_margin.detach().item()),
        "ref_full_margin": float(ref_full_margin.detach().item()),
        "policy_critical_margin": float(policy_critical_margin.detach().item()),
        "ref_critical_margin": float(ref_critical_margin.detach().item()),
        "policy_full_correct": float(policy_full_margin.detach().item() > 0.0),
        "ref_full_correct": float(ref_full_margin.detach().item() > 0.0),
        "policy_critical_correct": float(policy_critical_margin.detach().item() > 0.0),
        "ref_critical_correct": float(ref_critical_margin.detach().item() > 0.0),
    }

def evaluate_flip2_preference_loader(
    model,
    ref_model,
    flip2_val_rows,
    device,
    args,
    logger=None,
    pair_csv_path=None,
    pair_csv_fields=None,
    epoch=None,
):
    model.eval()
    ref_model.eval()

    total_pairs = 0
    skipped_zero_pair_batches = 0
    total_sppo_loss = 0.0

    policy_full_correct = 0.0
    ref_full_correct = 0.0
    policy_critical_correct = 0.0
    ref_critical_correct = 0.0

    policy_full_margin_sum = 0.0
    ref_full_margin_sum = 0.0
    policy_critical_margin_sum = 0.0
    ref_critical_margin_sum = 0.0

    target_gaps = []
    policy_full_margins = []
    ref_full_margins = []
    policy_critical_margins = []
    ref_critical_margins = []
    val_pair_rows = []

    num_eval_steps = max(1, len(flip2_val_rows) // args.flip2_minibatch_size)

    with torch.no_grad():
        for _ in range(num_eval_steps):
            minibatch_rows = random.sample(
                flip2_val_rows,
                k=min(args.flip2_minibatch_size, len(flip2_val_rows)),
            )

            pair_items = build_proteinmpnn_preference_pairs_from_batch(
                minibatch_rows,
                threshold=args.pair_threshold,
                max_pairs=args.flip2_minibatch_size,
            )

            if len(pair_items) == 0:
                skipped_zero_pair_batches += 1
                continue

            for pair in pair_items:
                pdb_id = pair["pdb_id"]
                positive_seq = pair["pos_sequence"]
                negative_seq = pair["neg_sequence"]

                batch_pos = make_batch_from_pdb(
                    pdb_id,
                    seq_override=positive_seq,
                    data_path=args.train_pos_seq_coord_path,
                )

                X, S_pos, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                    batch_pos,
                    device,
                )

                batch_neg = make_batch_from_pdb(
                    pdb_id,
                    seq_override=negative_seq,
                    data_path=args.train_pos_seq_coord_path,
                )

                _, S_neg, _, _, _, _, _, _ = featurize(batch_neg, device)

                mask_for_loss = chain_M * mask

                X_noisy = add_backbone_noise_to_X(
                    X,
                    args.flip2_backbone_noise,
                )

                logp_pos, _ = model_log_prob_per_res(
                    model,
                    X_noisy,
                    S_pos,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                )

                logp_neg, _ = model_log_prob_per_res(
                    model,
                    X_noisy,
                    S_neg,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                )

                logp_ref_pos, _ = model_log_prob_per_res(
                    ref_model,
                    X_noisy,
                    S_pos,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                )

                logp_ref_neg, _ = model_log_prob_per_res(
                    ref_model,
                    X_noisy,
                    S_neg,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                )

                sppo_loss = preference_sppo_loss(
                    logp_pos,
                    logp_neg,
                    logp_ref_pos,
                    logp_ref_neg,
                    S_pos,
                    S_neg,
                    mask_for_loss,
                    beta=args.beta,
                )

                m = compute_pair_metrics(
                    logp_pos,
                    logp_neg,
                    logp_ref_pos,
                    logp_ref_neg,
                    S_pos,
                    S_neg,
                    mask_for_loss,
                )
                val_pair_rows.append({
                    "epoch": epoch,
                    "step": "",
                    "global_step": "",
                    "split": "val",
                    "pair_id": pair["pair_id"],
                    "neg_source": pair.get("neg_source", "on_the_fly"),
                    "pos_sequence_id": pair.get("pos_sequence_id", ""),
                    "neg_sequence_id": pair.get("neg_sequence_id", ""),
                    "pos_target": pair["pos_target"],
                    "neg_target": pair["neg_target"],
                    "target_gap": pair["target_gap"],
                    "policy_full_margin": m["policy_full_margin"],
                    "ref_full_margin": m["ref_full_margin"],
                    "policy_critical_margin": m["policy_critical_margin"],
                    "ref_critical_margin": m["ref_critical_margin"],
                    "policy_full_correct": m["policy_full_correct"],
                    "ref_full_correct": m["ref_full_correct"],
                    "policy_critical_correct": m["policy_critical_correct"],
                    "ref_critical_correct": m["ref_critical_correct"],
                })

                total_pairs += 1
                total_sppo_loss += float(sppo_loss.detach().item())

                policy_full_correct += m["policy_full_correct"]
                ref_full_correct += m["ref_full_correct"]
                policy_critical_correct += m["policy_critical_correct"]
                ref_critical_correct += m["ref_critical_correct"]

                policy_full_margin_sum += m["policy_full_margin"]
                ref_full_margin_sum += m["ref_full_margin"]
                policy_critical_margin_sum += m["policy_critical_margin"]
                ref_critical_margin_sum += m["ref_critical_margin"]

                target_gaps.append(pair["target_gap"])
                policy_full_margins.append(m["policy_full_margin"])
                ref_full_margins.append(m["ref_full_margin"])
                policy_critical_margins.append(m["policy_critical_margin"])
                ref_critical_margins.append(m["ref_critical_margin"])

    denom = max(total_pairs, 1)

    metrics = {
        "val_pairs": total_pairs,
        "val_skipped_zero_pair_batches": skipped_zero_pair_batches,
        "val_sppo_loss": total_sppo_loss / denom,

        "val_policy_full_accuracy": policy_full_correct / denom,
        "val_ref_full_accuracy": ref_full_correct / denom,
        "val_policy_critical_accuracy": policy_critical_correct / denom,
        "val_ref_critical_accuracy": ref_critical_correct / denom,

        "val_policy_full_margin": policy_full_margin_sum / denom,
        "val_ref_full_margin": ref_full_margin_sum / denom,
        "val_policy_critical_margin": policy_critical_margin_sum / denom,
        "val_ref_critical_margin": ref_critical_margin_sum / denom,

        "val_full_margin_improvement":
            (policy_full_margin_sum - ref_full_margin_sum) / denom,

        "val_critical_margin_improvement":
            (policy_critical_margin_sum - ref_critical_margin_sum) / denom,

        "val_policy_full_pearson":
            safe_pearson(target_gaps, policy_full_margins),

        "val_ref_full_pearson":
            safe_pearson(target_gaps, ref_full_margins),

        "val_policy_critical_pearson":
            safe_pearson(target_gaps, policy_critical_margins),

        "val_ref_critical_pearson":
            safe_pearson(target_gaps, ref_critical_margins),
    }
    if pair_csv_path is not None and val_pair_rows:
        with open(pair_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pair_csv_fields)
            writer.writerows(val_pair_rows)

    return metrics

def dataframe_to_flip2_rows(df, pdb_id):
    rows = []

    for i, row in df.iterrows():
        seq = str(row["sequence"]).strip()
        if not seq:
            continue

        rows.append({
            "sequence_id": str(row["sequence_id"]) if "sequence_id" in df.columns else str(i),
            "pdb_id": pdb_id,
            "sequence": seq,
            "target": float(row["target"]),
        })

    return rows

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    import os
    import random
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import logging
    from concurrent.futures import ProcessPoolExecutor    
   

    log_dir = os.path.join(args.path_for_outputs, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    dpo_csv_path = args.csv_path_train  # or hardcode path if needed
    dpo_df = pd.read_csv(dpo_csv_path)


    if "set" in dpo_df.columns and "validation" in dpo_df.columns:

        val_mask = dpo_df["validation"].apply(parse_boolish)
        set_col = dpo_df["set"].astype(str).str.lower()

        flip2_train_df = dpo_df[
            (set_col == "train") & (~val_mask)
        ].copy()

        flip2_val_df = dpo_df[
            (set_col == "train") & (val_mask)
        ].copy()

    elif "split" in dpo_df.columns:

        split_col = dpo_df["split"].astype(str).str.lower()

        flip2_train_df = dpo_df[
            split_col == "train"
        ].copy()

        flip2_val_df = dpo_df[
            split_col.isin(["validation", "valid", "val"])
        ].copy()

    else:
        raise ValueError(
            "CSV must contain either columns: sequence,set,validation,target "
            "or columns: sequence,target,split"
        )

        
    flip2_train_rows = dataframe_to_flip2_rows(
        flip2_train_df,
        args.flip2_pdb_id,
    )

    flip2_val_rows = dataframe_to_flip2_rows(
        flip2_val_df,
        args.flip2_pdb_id,
    )
    positive_pool = PositiveSequencePool(
        csv_file=args.positive_pool_csv,
        positive_threshold=args.positive_threshold,
    )

    negative_pool = NegativeSequencePool(
        csv_file=args.real_negative_csv,
    )



    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded Flip2 rows for on-the-fly pairs: {len(flip2_train_rows)}")

    ce_df = pd.read_csv(args.ce_csv_path_train)
    
    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint
    pair_log_dir = os.path.join(base_folder, "pair_logs")
    os.makedirs(pair_log_dir, exist_ok=True)

    train_pair_csv_path = os.path.join(pair_log_dir, "train_pairs_used.csv")
    val_pair_csv_path = os.path.join(pair_log_dir, "val_pairs_used.csv")

    pair_csv_fields = [
        "epoch",
        "step",
        "global_step",
        "split",
        "pair_id",
        "neg_source",
        "pos_sequence_id",
        "neg_sequence_id",
        "pos_target",
        "neg_target",
        "target_gap",
        "policy_full_margin",
        "ref_full_margin",
        "policy_critical_margin",
        "ref_critical_margin",
        "policy_full_correct",
        "ref_full_correct",
        "policy_critical_correct",
        "ref_critical_correct",
    ]

    for csv_path in [train_pair_csv_path, val_pair_csv_path]:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pair_csv_fields)
            writer.writeheader()

    logger.info(f"Train pair CSV: {train_pair_csv_path}")
    logger.info(f"Val pair CSV: {val_pair_csv_path}")
   
    if args.debug:
        args.num_examples_per_epoch = 100000
        args.max_protein_length = 10000
        args.batch_size = 1000


    logger.info("Creating ProteinMPNN model")
    model = ProteinMPNN(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_encoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
    )
    model.to(device)
    base_lr = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)


    vanilla_path = "/dapustor/nilufer/ProteinMPNN/vanilla_model_weights/v_48_020.pt"

    logger.info(f"Loading vanilla checkpoint from: {vanilla_path}")
    checkpoint = torch.load(vanilla_path, map_location="cpu")
    logger.info("Vanilla checkpoint loaded")

    model.load_state_dict(checkpoint["model_state_dict"])
    ref_model = copy.deepcopy(model)

    # If model was wrapped with DataParallel, get underlying module
    try:
        ref_model = ref_model.to(device)
    except Exception:
        # ensure module is on device (for DataParallel)
        if hasattr(ref_model, 'module'):
            ref_model.module.to(device)
        else:
            ref_model.to(device)

    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()


    if PATH:
        checkpoint = torch.load(PATH, map_location="cpu")
        print(f"Loaded checkpoint keys: {checkpoint.keys()}")

        # Load model weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model_state_dict from checkpoint.")
        else:
            model.load_state_dict(checkpoint)  # direct load if it’s a plain state_dict
            print("Loaded checkpoint as plain state_dict.")

        # Initialize training state manually
        total_step = 0
        epoch = 0
    else:
        print("No checkpoint provided, starting fresh.")
        total_step = 0
        epoch = 0

    


    if PATH:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    with ProcessPoolExecutor(max_workers=4) as executor:
        
        steps_per_epoch = math.ceil(
            len(flip2_train_rows) / args.flip2_minibatch_size
        )        

        num_training_steps = max(1, args.num_epochs * steps_per_epoch)
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    

        reload_c = 0 
        steps_per_epoch = math.ceil(
            len(flip2_train_rows) / args.flip2_minibatch_size
        )        
        best_val_score = float("-inf")
        for e in range(args.num_epochs):
            results_train_per_epoch = []  # For training
            results_valid_per_epoch = []
            t0 = time.time()
            e = epoch + e
            model.train()
            epoch_loss = 0.0 
            progress_bar = range(steps_per_epoch)
            train_sum, train_weights = 0., 0.
            train_critic_site_count = 0.
            train_critic_site_weights = 0.
            epoch_sppo_loss_sum = 0.0
            epoch_ce_loss_sum = 0.0
            epoch_total_loss_sum = 0.0

            epoch_policy_full_correct = 0.0
            epoch_ref_full_correct = 0.0
            epoch_policy_critical_correct = 0.0
            epoch_ref_critical_correct = 0.0

            epoch_policy_full_margin_sum = 0.0
            epoch_ref_full_margin_sum = 0.0
            epoch_policy_critical_margin_sum = 0.0
            epoch_ref_critical_margin_sum = 0.0

            epoch_target_gaps = []
            epoch_policy_full_margins = []
            epoch_ref_full_margins = []
            epoch_policy_critical_margins = []
            epoch_ref_critical_margins = []

            epoch_pair_count = 0
            for step in progress_bar:
                optimizer.zero_grad()

           
                onthefly_pair_items = []

                while len(onthefly_pair_items) == 0:
                    minibatch_rows = random.sample(
                        flip2_train_rows,
                        k=min(args.flip2_minibatch_size, len(flip2_train_rows)),
                    )

                    onthefly_pair_items = build_proteinmpnn_preference_pairs_from_batch(
                        minibatch_rows,
                        threshold=args.pair_threshold,
                        max_pairs=args.onthefly_pairs_per_step,
                    )

                real_pair_items = build_random_real_pairs(
                    positive_pool=positive_pool,
                    negative_pool=negative_pool,
                    num_real_pairs=args.real_pairs_per_step,
                    pdb_id=args.flip2_pdb_id,
                )
                
                pair_items = onthefly_pair_items + real_pair_items
                random.shuffle(pair_items)
                
                batch_sppo_loss = 0.0
                batch_critical_correct = 0.0
                batch_pair_count = 0
                batch_policy_full_correct = 0.0
                batch_ref_full_correct = 0.0
                batch_policy_critical_correct = 0.0
                batch_ref_critical_correct = 0.0

                batch_policy_full_margin_sum = 0.0
                batch_ref_full_margin_sum = 0.0
                batch_policy_critical_margin_sum = 0.0
                batch_ref_critical_margin_sum = 0.0

                batch_target_gaps = []
                batch_policy_full_margins = []
                batch_ref_full_margins = []
                batch_policy_critical_margins = []
                batch_ref_critical_margins = []

                train_pair_rows = []
                source_metrics = {}
                for pair in pair_items:
                    pdb_id = pair["pdb_id"]
                    positive_seq = pair["pos_sequence"]
                    negative_seq = pair["neg_sequence"]

                    batch_pos = make_batch_from_pdb(
                        pdb_id,
                        visible_chains=[],
                        masked_chains=[],
                        seq_override=positive_seq,
                        data_path=args.train_pos_seq_coord_path,
                    )

                    X, S_pos, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                        batch_pos,
                        device,
                    )

                    batch_neg = make_batch_from_pdb(
                        pdb_id,
                        visible_chains=[],
                        masked_chains=[],
                        seq_override=negative_seq,
                        data_path=args.train_pos_seq_coord_path,
                    )

                    _, S_neg, _, _, _, _, _, _ = featurize(
                        batch_neg,
                        device,
                    )

                    mask_for_loss = chain_M * mask

                    X_flip2_noisy = add_backbone_noise_to_X(
                        X,
                        args.flip2_backbone_noise,
                    )

                    logp_pos, _ = model_log_prob_per_res(
                        model,
                        X_flip2_noisy,
                        S_pos,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                    )

                    logp_neg, _ = model_log_prob_per_res(
                        model,
                        X_flip2_noisy,
                        S_neg,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                    )

                    with torch.no_grad():
                        logp_ref_pos, _ = model_log_prob_per_res(
                            ref_model,
                            X_flip2_noisy,
                            S_pos,
                            mask,
                            chain_M,
                            residue_idx,
                            chain_encoding_all,
                        )

                        logp_ref_neg, _ = model_log_prob_per_res(
                            ref_model,
                            X_flip2_noisy,
                            S_neg,
                            mask,
                            chain_M,
                            residue_idx,
                            chain_encoding_all,
                        )

                    pair_sppo_loss = preference_sppo_loss(
                        logp_pos,
                        logp_neg,
                        logp_ref_pos,
                        logp_ref_neg,
                        S_pos,
                        S_neg,
                        mask_for_loss,
                        beta=args.beta,
                    )
                    m = compute_pair_metrics(
                        logp_pos,
                        logp_neg,
                        logp_ref_pos,
                        logp_ref_neg,
                        S_pos,
                        S_neg,
                        mask_for_loss,
                    )
                    src = pair.get("neg_source", "on_the_fly")

                    if src not in source_metrics:
                        source_metrics[src] = {
                            "count": 0,
                            "policy_full_correct": 0.0,
                            "ref_full_correct": 0.0,
                            "policy_critical_correct": 0.0,
                            "ref_critical_correct": 0.0,
                            "policy_full_margin_sum": 0.0,
                            "ref_full_margin_sum": 0.0,
                            "policy_critical_margin_sum": 0.0,
                            "ref_critical_margin_sum": 0.0,
                        }

                    source_metrics[src]["count"] += 1
                    source_metrics[src]["policy_full_correct"] += m["policy_full_correct"]
                    source_metrics[src]["ref_full_correct"] += m["ref_full_correct"]
                    source_metrics[src]["policy_critical_correct"] += m["policy_critical_correct"]
                    source_metrics[src]["ref_critical_correct"] += m["ref_critical_correct"]
                    source_metrics[src]["policy_full_margin_sum"] += m["policy_full_margin"]
                    source_metrics[src]["ref_full_margin_sum"] += m["ref_full_margin"]
                    source_metrics[src]["policy_critical_margin_sum"] += m["policy_critical_margin"]
                    source_metrics[src]["ref_critical_margin_sum"] += m["ref_critical_margin"]
                    batch_policy_full_correct += m["policy_full_correct"]
                    batch_ref_full_correct += m["ref_full_correct"]
                    batch_policy_critical_correct += m["policy_critical_correct"]
                    batch_ref_critical_correct += m["ref_critical_correct"]

                    batch_policy_full_margin_sum += m["policy_full_margin"]
                    batch_ref_full_margin_sum += m["ref_full_margin"]
                    batch_policy_critical_margin_sum += m["policy_critical_margin"]
                    batch_ref_critical_margin_sum += m["ref_critical_margin"]

                    batch_target_gaps.append(pair["target_gap"])
                    batch_policy_full_margins.append(m["policy_full_margin"])
                    batch_ref_full_margins.append(m["ref_full_margin"])
                    batch_policy_critical_margins.append(m["policy_critical_margin"])
                    batch_ref_critical_margins.append(m["ref_critical_margin"])

                    train_pair_rows.append({
                        "epoch": e + 1,
                        "step": step + 1,
                        "global_step": total_step,
                        "split": "train",
                        "pair_id": pair["pair_id"],
                        "neg_source": pair.get("neg_source", "on_the_fly"),
                        "pos_sequence_id": pair.get("pos_sequence_id", ""),
                        "neg_sequence_id": pair.get("neg_sequence_id", ""),
                        "pos_target": pair["pos_target"],
                        "neg_target": pair["neg_target"],
                        "target_gap": pair["target_gap"],
                        "policy_full_margin": m["policy_full_margin"],
                        "ref_full_margin": m["ref_full_margin"],
                        "policy_critical_margin": m["policy_critical_margin"],
                        "ref_critical_margin": m["ref_critical_margin"],
                        "policy_full_correct": m["policy_full_correct"],
                        "ref_full_correct": m["ref_full_correct"],
                        "policy_critical_correct": m["policy_critical_correct"],
                        "ref_critical_correct": m["ref_critical_correct"],
                    })
                    epoch_policy_full_correct += m["policy_full_correct"]
                    epoch_ref_full_correct += m["ref_full_correct"]
                    epoch_policy_critical_correct += m["policy_critical_correct"]
                    epoch_ref_critical_correct += m["ref_critical_correct"]

                    epoch_policy_full_margin_sum += m["policy_full_margin"]
                    epoch_ref_full_margin_sum += m["ref_full_margin"]
                    epoch_policy_critical_margin_sum += m["policy_critical_margin"]
                    epoch_ref_critical_margin_sum += m["ref_critical_margin"]

                    epoch_target_gaps.append(pair["target_gap"])

                    epoch_policy_full_margins.append(
                        m["policy_full_margin"]
                    )

                    epoch_ref_full_margins.append(
                        m["ref_full_margin"]
                    )

                    epoch_policy_critical_margins.append(
                        m["policy_critical_margin"]
                    )

                    epoch_ref_critical_margins.append(
                        m["ref_critical_margin"]
                    )

                    epoch_pair_count += 1

                    mask_diff = (S_pos != S_neg).float() * mask_for_loss

                    critical_correct = (
                        torch.sum(logp_pos * mask_diff, dim=1)
                        >
                        torch.sum(logp_neg * mask_diff, dim=1)
                    ).float()

                    batch_sppo_loss = batch_sppo_loss + pair_sppo_loss
                    batch_critical_correct += critical_correct.item()
                    batch_pair_count += 1

                sppo_loss = batch_sppo_loss / max(batch_pair_count, 1)
                batch_denom = max(batch_pair_count, 1)

                batch_metrics = {
                    "policy_full_accuracy": batch_policy_full_correct / batch_denom,
                    "ref_full_accuracy": batch_ref_full_correct / batch_denom,
                    "policy_critical_accuracy": batch_policy_critical_correct / batch_denom,
                    "ref_critical_accuracy": batch_ref_critical_correct / batch_denom,

                    "policy_full_margin": batch_policy_full_margin_sum / batch_denom,
                    "ref_full_margin": batch_ref_full_margin_sum / batch_denom,
                    "policy_critical_margin": batch_policy_critical_margin_sum / batch_denom,
                    "ref_critical_margin": batch_ref_critical_margin_sum / batch_denom,

                    "full_margin_improvement":
                        (batch_policy_full_margin_sum - batch_ref_full_margin_sum) / batch_denom,

                    "critical_margin_improvement":
                        (batch_policy_critical_margin_sum - batch_ref_critical_margin_sum) / batch_denom,

                    "policy_full_pearson":
                        safe_pearson(batch_target_gaps, batch_policy_full_margins),

                    "ref_full_pearson":
                        safe_pearson(batch_target_gaps, batch_ref_full_margins),

                    "policy_critical_pearson":
                        safe_pearson(batch_target_gaps, batch_policy_critical_margins),

                    "ref_critical_pearson":
                        safe_pearson(batch_target_gaps, batch_ref_critical_margins),
                }

                train_critic_site_count += batch_critical_correct
                train_critic_site_weights += batch_pair_count

                # 2. Original dataset CE loss
                ce_total_loss = 0.0
                ce_count = 0

                ce_rows = ce_df.sample(
                    n=min(args.ce_batch_size, len(ce_df)),
                    replace=False,
                )

                for _, ce_row in ce_rows.iterrows():
                    ce_pdb_id = ce_row["Sequence_ID"]
                    ce_seq = ce_row["GT_sequence"]

                    ce_batch = make_batch_from_pdb(
                        ce_pdb_id,
                        seq_override=ce_seq,
                        data_path=args.ce_pos_seq_coord_path,
                    )

                    X_ce, S_ce, mask_ce, lengths_ce, chain_M_ce, residue_idx_ce, mask_self_ce, chain_encoding_ce = featurize(
                        ce_batch,
                        device,
                    )

                    mask_ce_loss = chain_M_ce * mask_ce


                    _, logp_ce_all = model_log_prob_per_res(
                        model,
                        X_ce,
                        S_ce,
                        mask_ce,
                        chain_M_ce,
                        residue_idx_ce,
                        chain_encoding_ce,
                    )

                    ce_item_loss = ce_nll_loss(
                        S_ce,
                        logp_ce_all,
                        mask_ce_loss,
                    )

                    ce_total_loss = ce_total_loss + ce_item_loss
                    ce_count += 1

                ce_loss = ce_total_loss / max(ce_count, 1)

                # 3. Final combined loss
                loss = sppo_loss + args.lambda_ce * ce_loss
                epoch_sppo_loss_sum += float(sppo_loss.detach().item())
                epoch_ce_loss_sum += float(ce_loss.detach().item())
                epoch_total_loss_sum += float(loss.detach().item())

                # 4. Backward
                scaler.scale(loss).backward()
            
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_step += 1
                if (step + 1) % args.log_every == 0:
                    batch_policy_critical_acc = batch_critical_correct / max(batch_pair_count, 1)

                    logger.info(
                        f"Train minibatch | "
                        f"epoch={e+1} | "
                        f"step={step+1}/{steps_per_epoch} | "
                        f"global_step={total_step} | "
                        f"pairs={batch_pair_count} | "
                        f"ce_samples={ce_count} | "
                        f"total_loss={loss.item():.6f} | "
                        f"sppo_loss={sppo_loss.item():.6f} | "
                        f"ce_loss={ce_loss.item():.6f} | "
                        f"policy_full_acc={batch_metrics['policy_full_accuracy']:.6f} | "
                        f"ref_full_acc={batch_metrics['ref_full_accuracy']:.6f} | "
                        f"policy_critical_acc={batch_metrics['policy_critical_accuracy']:.6f} | "
                        f"ref_critical_acc={batch_metrics['ref_critical_accuracy']:.6f} | "
                        f"policy_full_margin={batch_metrics['policy_full_margin']:.6f} | "
                        f"ref_full_margin={batch_metrics['ref_full_margin']:.6f} | "
                        f"full_margin_improve={batch_metrics['full_margin_improvement']:.6f} | "
                        f"policy_critical_margin={batch_metrics['policy_critical_margin']:.6f} | "
                        f"ref_critical_margin={batch_metrics['ref_critical_margin']:.6f} | "
                        f"critical_margin_improve={batch_metrics['critical_margin_improvement']:.6f} | "
                        f"policy_full_pearson={batch_metrics['policy_full_pearson']:.6f} | "
                        f"ref_full_pearson={batch_metrics['ref_full_pearson']:.6f} | "
                        f"policy_critical_pearson={batch_metrics['policy_critical_pearson']:.6f} | "
                        f"ref_critical_pearson={batch_metrics['ref_critical_pearson']:.6f}"
                    )
                    for src, sm in source_metrics.items():
                        n = max(sm["count"], 1)

                        logger.info(
                            f"Train minibatch source={src} | "
                            f"epoch={e+1} | "
                            f"step={step+1}/{steps_per_epoch} | "
                            f"global_step={total_step} | "
                            f"pairs={sm['count']} | "
                            f"policy_full_acc={sm['policy_full_correct'] / n:.6f} | "
                            f"ref_full_acc={sm['ref_full_correct'] / n:.6f} | "
                            f"policy_critical_acc={sm['policy_critical_correct'] / n:.6f} | "
                            f"ref_critical_acc={sm['ref_critical_correct'] / n:.6f} | "
                            f"policy_full_margin={sm['policy_full_margin_sum'] / n:.6f} | "
                            f"ref_full_margin={sm['ref_full_margin_sum'] / n:.6f} | "
                            f"full_margin_improve={(sm['policy_full_margin_sum'] - sm['ref_full_margin_sum']) / n:.6f} | "
                            f"policy_critical_margin={sm['policy_critical_margin_sum'] / n:.6f} | "
                            f"ref_critical_margin={sm['ref_critical_margin_sum'] / n:.6f} | "
                            f"critical_margin_improve={(sm['policy_critical_margin_sum'] - sm['ref_critical_margin_sum']) / n:.6f}"
                        )
                if train_pair_rows:
                    with open(train_pair_csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=pair_csv_fields)
                        writer.writerows(train_pair_rows)
               
                epoch_loss += loss.item()

                train_sum += loss.item()
                train_weights += torch.sum(mask).cpu().data.numpy()

            avg_train_loss = epoch_loss / steps_per_epoch
            elapsed = time.time() - t0
            avg_train_loss = epoch_loss / steps_per_epoch
            elapsed = time.time() - t0
            denom_pairs = max(epoch_pair_count, 1)
            denom_steps = max(steps_per_epoch, 1)

            train_metrics = {
                "train_total_loss": epoch_total_loss_sum / denom_steps,
                "train_sppo_loss": epoch_sppo_loss_sum / denom_steps,
                "train_ce_loss": epoch_ce_loss_sum / denom_steps,

                "train_policy_full_accuracy": epoch_policy_full_correct / denom_pairs,
                "train_ref_full_accuracy": epoch_ref_full_correct / denom_pairs,
                "train_policy_critical_accuracy": epoch_policy_critical_correct / denom_pairs,
                "train_ref_critical_accuracy": epoch_ref_critical_correct / denom_pairs,

                "train_policy_full_margin": epoch_policy_full_margin_sum / denom_pairs,
                "train_ref_full_margin": epoch_ref_full_margin_sum / denom_pairs,
                "train_policy_critical_margin": epoch_policy_critical_margin_sum / denom_pairs,
                "train_ref_critical_margin": epoch_ref_critical_margin_sum / denom_pairs,

                "train_full_margin_improvement":
                    (epoch_policy_full_margin_sum - epoch_ref_full_margin_sum) / denom_pairs,

                "train_critical_margin_improvement":
                    (epoch_policy_critical_margin_sum - epoch_ref_critical_margin_sum) / denom_pairs,

                "train_policy_full_pearson":
                    safe_pearson(epoch_target_gaps, epoch_policy_full_margins),

                "train_ref_full_pearson":
                    safe_pearson(epoch_target_gaps, epoch_ref_full_margins),

                "train_policy_critical_pearson":
                    safe_pearson(epoch_target_gaps, epoch_policy_critical_margins),

                "train_ref_critical_pearson":
                    safe_pearson(epoch_target_gaps, epoch_ref_critical_margins),
            }
            logger.info(
                f"Train epoch {e+1} | "
                f"total_loss={train_metrics['train_total_loss']:.6f} | "
                f"sppo_loss={train_metrics['train_sppo_loss']:.6f} | "
                f"ce_loss={train_metrics['train_ce_loss']:.6f} | "
                f"policy_full_acc={train_metrics['train_policy_full_accuracy']:.6f} | "
                f"ref_full_acc={train_metrics['train_ref_full_accuracy']:.6f} | "
                f"policy_critical_acc={train_metrics['train_policy_critical_accuracy']:.6f} | "
                f"ref_critical_acc={train_metrics['train_ref_critical_accuracy']:.6f} | "
                f"policy_full_margin={train_metrics['train_policy_full_margin']:.6f} | "
                f"ref_full_margin={train_metrics['train_ref_full_margin']:.6f} | "
                f"full_margin_improve={train_metrics['train_full_margin_improvement']:.6f} | "
                f"policy_critical_margin={train_metrics['train_policy_critical_margin']:.6f} | "
                f"ref_critical_margin={train_metrics['train_ref_critical_margin']:.6f} | "
                f"critical_margin_improve={train_metrics['train_critical_margin_improvement']:.6f} | "
                f"policy_full_pearson={train_metrics['train_policy_full_pearson']:.6f} | "
                f"ref_full_pearson={train_metrics['train_ref_full_pearson']:.6f} | "
                f"policy_critical_pearson={train_metrics['train_policy_critical_pearson']:.6f} | "
                f"ref_critical_pearson={train_metrics['train_ref_critical_pearson']:.6f}"
            )

            
            val_metrics = evaluate_flip2_preference_loader(
                model,
                ref_model,
                flip2_val_rows,
                device,
                args,
                logger=logger,
                pair_csv_path=val_pair_csv_path,
                pair_csv_fields=pair_csv_fields,
                epoch=e + 1,
            )

            logger.info(
                f"Validation epoch {e+1} | "
                f"pairs={val_metrics['val_pairs']} | "
                f"skipped_zero_pair_batches={val_metrics['val_skipped_zero_pair_batches']} | "
                f"val_sppo_loss={val_metrics['val_sppo_loss']:.6f} | "
                f"val_policy_full_acc={val_metrics['val_policy_full_accuracy']:.6f} | "
                f"val_ref_full_acc={val_metrics['val_ref_full_accuracy']:.6f} | "
                f"val_policy_critical_acc={val_metrics['val_policy_critical_accuracy']:.6f} | "
                f"val_ref_critical_acc={val_metrics['val_ref_critical_accuracy']:.6f} | "
                f"val_policy_full_margin={val_metrics['val_policy_full_margin']:.6f} | "
                f"val_ref_full_margin={val_metrics['val_ref_full_margin']:.6f} | "
                f"val_full_margin_improve={val_metrics['val_full_margin_improvement']:.6f} | "
                f"val_policy_critical_margin={val_metrics['val_policy_critical_margin']:.6f} | "
                f"val_ref_critical_margin={val_metrics['val_ref_critical_margin']:.6f} | "
                f"val_critical_margin_improve={val_metrics['val_critical_margin_improvement']:.6f} | "
                f"val_policy_full_pearson={val_metrics['val_policy_full_pearson']:.6f} | "
                f"val_ref_full_pearson={val_metrics['val_ref_full_pearson']:.6f} | "
                f"val_policy_critical_pearson={val_metrics['val_policy_critical_pearson']:.6f} | "
                f"val_ref_critical_pearson={val_metrics['val_ref_critical_pearson']:.6f}"
            )
            current_val_score = val_metrics["val_policy_critical_accuracy"]

            if current_val_score > best_val_score:
                best_val_score = current_val_score
                save_path = os.path.join(base_folder, "model_weights", "best.pt")
                torch.save({
                    "epoch": e + 1,
                    "step": total_step,
                    "num_edges": args.num_neighbors,
                    "noise_level": args.flip2_backbone_noise,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_score": best_val_score,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                }, save_path)

                logger.info(
                    f"Saved new best model to {save_path} "
                    f"(val_policy_critical_accuracy={best_val_score:.6f})"
                )
            model.train()

            
            
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename)


if __name__ == "__main__":  
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #argparser.add_argument("--path_for_training_data", type=str, default="/data/datasets/nilufer/CAPEMPNN/CAPE_MPNN/data/input/CAPE-MPNN/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./Flip2", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=2, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.02, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--csv_path_train", type=str, default="/dapustor/nilufer/ProteinMPNN/flip2_data/imine_reductase_two_to_many.csv", help="CSV with training positive/negative sequence pairs")
    argparser.add_argument("--train_pos_seq_coord_path", type=str, default="/dapustor/nilufer/ProteinMPNN/flip2_data/structure", help="path for loading training data") 
    argparser.add_argument("--val_pos_seq_coord_path", type=str, default="/dapustor/nilufer/YYH/ProteinMPNN/data/val_pos_seq_coords_with_mask", help="path for loading training data") 
    argparser.add_argument("--beta", type=float, default=0.2)
    argparser.add_argument("--lambda_ce", type=float, default=0.2)

    argparser.add_argument("--flip2_backbone_noise", type=float, default=0.0)

    argparser.add_argument("--ce_csv_path_train", type=str, default="/dapustor/nilufer/YYH/ProteinMPNN/data/train_protein_mpnn_final.csv", help="CSV with training positive/negative sequence pairs for contrastive learning")
    argparser.add_argument("--ce_pos_seq_coord_path", type=str, default="/dapustor/nilufer/YYH/ProteinMPNN/data/train_pos_seq_coords_with_mask", help="path for loading contrastive learning training data")
          
    argparser.add_argument("--flip2_minibatch_size", type=int, default=10)
    argparser.add_argument("--max_pairs_per_step", type=int, default=10)
    argparser.add_argument("--ce_batch_size", type=int, default=1)
    argparser.add_argument("--pair_threshold", type=float, default=1)
    argparser.add_argument("--flip2_pdb_id", type=str, default="5OCM_B")
    argparser.add_argument("--log_every", type=int, default=1)
    argparser.add_argument("--real_pairs_per_step", type=int, default=2)
    argparser.add_argument("--onthefly_pairs_per_step", type=int, default=8)
    argparser.add_argument("--positive_pool_csv",type=str,default="/dapustor/nilufer/ProteinMPNN/flip2_data/imine_reductase_three_to_many.csv", )
    argparser.add_argument("--real_negative_csv",type=str,default="/dapustor/nilufer/ProteinMPNN/flip2_data/inactive_seq_three_mutations.csv",)
    argparser.add_argument("--positive_threshold",type=float,default=0.5)
    argparser.add_argument("--num_real_pairs",type=int,default=3,)
    args = argparser.parse_args()
    main(args)