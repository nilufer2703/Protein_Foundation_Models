import pandas as pd
import numpy as np
import math

# Input files
negative_csv = "/dapustor/nilufer/ProteinMPNN/flip2_data/WT_neg_recombinant_sequences.csv"
positive_csv = "/dapustor/nilufer/ProteinMPNN/flip2_data/csvs/imine_reductase_four_to_many.csv"

# Output file
output_csv = "/dapustor/nilufer/ProteinMPNN/flip2_data/pairs_4_to_Many_WT_neg_recombinanat.csv"

# Reproducibility
random_seed = 42
rng = np.random.default_rng(random_seed)

# Read CSV files
neg_df = pd.read_csv(negative_csv)
pos_df = pd.read_csv(positive_csv)

# Get negative sequences
negative_sequences = (
    neg_df["sequence"]
    .dropna()
    .astype(str)
    .unique()
)


# Get positive sequences
if "set" in pos_df.columns:
    # Original format:
    # sequence,set,validation,target

    positive_sequences = (
        pos_df[
            (pos_df["set"] == "train") &
            (~pos_df["validation"]) &
            (pos_df["target"] > 0.5)
        ]["sequence"]
        .dropna()
        .astype(str)
        .unique()
    )

elif "split" in pos_df.columns:
    # New format:
    # sequence,target,num_mutations,split

    positive_sequences = (
        pos_df[
            (pos_df["split"].astype(str).str.lower() == "train") &
            (pos_df["target"] > 0.5)
        ]["sequence"]
        .dropna()
        .astype(str)
        .unique()
    )

else:
    raise ValueError(
        f"Could not find either 'set' or 'split' column. "
        f"Available columns: {list(pos_df.columns)}"
    )

# Print counts
print(f"Number of positive train sequences with target > 0.5: {len(positive_sequences)}")
print(f"Number of negative sequences: {len(negative_sequences)}")

if len(positive_sequences) == 0:
    raise ValueError("No positive train sequences found with target > 0.5")

if len(negative_sequences) == 0:
    raise ValueError("No negative sequences found")

# Randomize order
positive_sequences = rng.permutation(positive_sequences)
negative_sequences = rng.permutation(negative_sequences)

# Repeat positives to match number of negatives
repeats_needed = math.ceil(len(negative_sequences) / len(positive_sequences))
positive_repeated = np.tile(positive_sequences, repeats_needed)

# Trim exactly to negative count
positive_repeated = positive_repeated[:len(negative_sequences)]

# Shuffle positive assignments
positive_repeated = rng.permutation(positive_repeated)

# Create final dataframe
paired_df = pd.DataFrame({
    "Sequence_ID": [f"pair_{i+1}" for i in range(len(negative_sequences))],
    "Positive_sequence": positive_repeated,
    "Negative_sequence": negative_sequences
})

# Save CSV
paired_df.to_csv(output_csv, index=False)

print(f"\nSaved: {output_csv}")
print(f"Total pairs: {len(paired_df)}")
print("\nPreview:")
print(paired_df.head())