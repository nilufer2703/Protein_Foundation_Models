from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools


import logging
import numpy as np
import torch
import random



import logging
import numpy as np
import torch
import random      

import random
import pandas as pd


class PositiveSequencePool:
    def __init__(self, csv_file, positive_threshold=0.5):
        df = pd.read_csv(csv_file)

        if "sequence" not in df.columns:
            raise KeyError(f"'sequence' column not found in {csv_file}")

        if "target" not in df.columns:
            raise KeyError(f"'target' column not found in {csv_file}")

        # Keep only training rows
        if "split" in df.columns:
            # New format:
            # sequence,target,num_mutations,split
            df = df[
                df["split"].astype(str).str.lower() == "train"
            ].copy()

        elif "set" in df.columns:
            # Original format:
            # sequence,set,validation,target
            df = df[
                (df["set"].astype(str).str.lower() == "train") &
                (~df["validation"].astype(bool))
            ].copy()

        else:
            raise ValueError(
                f"Could not find either 'split' or 'set' column. "
                f"Available columns: {list(df.columns)}"
            )

        # Keep only positive sequences
        df = df[df["target"].astype(float) > positive_threshold].copy()

        if len(df) == 0:
            raise ValueError(
                f"No positive training sequences found with target > {positive_threshold}"
            )

        self.sequences = df["sequence"].dropna().astype(str).tolist()
        self.targets = df["target"].astype(float).tolist()

    def sample(self, n):
        return [
            {
                "sequence": self.sequences[i],
                "target": self.targets[i],
            }
            for i in random.choices(range(len(self.sequences)), k=n)
        ]


class NegativeSequencePool:
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        if "sequence" not in df.columns:
            raise KeyError(f"'sequence' column not found in {csv_file}")

        self.sequences = df["sequence"].dropna().astype(str).tolist()

        if len(self.sequences) == 0:
            raise ValueError(f"No negative sequences found in {csv_file}")

    def sample(self, n):
        return random.choices(self.sequences, k=n)


def build_random_real_pairs(
    positive_pool,
    negative_pool,
    num_real_pairs,
    pdb_id,
):
    if num_real_pairs <= 0:
        return []

    positives = positive_pool.sample(num_real_pairs)
    negatives = negative_pool.sample(num_real_pairs)

    real_pairs = []

    for i, (pos_item, neg_seq) in enumerate(zip(positives, negatives)):

        pos_seq_id = pos_item.get("sequence_id", f"random_pos_{i}")
        neg_seq_id = f"random_real_neg_{i}"

        real_pairs.append({
            "pair_id": f"{pos_seq_id}__vs__{neg_seq_id}",
            "pdb_id": pdb_id,

            "pos_sequence": pos_item["sequence"],
            "neg_sequence": neg_seq,

            "pos_target": float(pos_item["target"]),
            "neg_target": 0.0,
            "target_gap": float(pos_item["target"]),

            "pos_sequence_id": pos_seq_id,
            "neg_sequence_id": neg_seq_id,

            "neg_source": "random_real_negative",
        })

    return real_pairs