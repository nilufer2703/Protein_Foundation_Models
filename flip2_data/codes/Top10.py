import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


input_csv = "/dapustor/nilufer/ProteinMPNN/flip2_data/csvs/imine_reductase_three_to_many.csv"
EXPERIMENT = "3_To_Many_WT_AT_In"

models = [
    {
        "name": "ProteinMPNN_base_020",
        "log_probs_csv": "/dapustor/nilufer/ProteinMPNN/flip2_data/base_02_output/log_probs/5OCM.csv",
    },
    {
        "name": "Inactive_seq_3_mutations",
        "log_probs_csv": "/dapustor/nilufer/ProteinMPNN/flip2_data/3_to_Many_Inactive_seq_3_mutations/log_probs/5OCM.csv",
    },
    {
        "name": "Inactive_seq_3_mutations_otf",
        "log_probs_csv": "/dapustor/nilufer/ProteinMPNN/flip2_data/3_to_Many_otf_inactive_sequence_3_mutations/log_probs/5OCM.csv",
    },
    
    {
        "name": "WT_neg_recombinant",
        "log_probs_csv": "/dapustor/nilufer/ProteinMPNN/flip2_data/3_to_Many_WT_neg_recombinant/log_probs/5OCM.csv",
    },
    {
        "name": "AT_neg_recombinant",
        "log_probs_csv": "/dapustor/nilufer/ProteinMPNN/flip2_data/3_to_Many_three_AT_neg_recombinant/log_probs/5OCM.csv",
    },
]

output_dir = Path(f"/dapustor/nilufer/jonathan/ProteinMPNN/Results/topk_overlap_analysis/{EXPERIMENT}")
output_dir.mkdir(parents=True, exist_ok=True)

output_overlap_csv = output_dir / "topk_overlap_summary_train_val_test.csv"
output_ranked_csv = output_dir / "topk_overlap_details_train_val_test.csv"

chain_b_start_row = 290
topk_values = [10, 100, 500, 1000]

aa_to_col = {aa: aa for aa in list("ARNDCQEGHILKMFPSTWYVX")}


def sequence_average_log_prob(sequence, log_probs_df, row_offset=0):
    sequence = str(sequence).strip()

    if row_offset + len(sequence) > len(log_probs_df):
        raise ValueError(
            f"Sequence with offset needs rows up to {row_offset + len(sequence) - 1}, "
            f"but log-prob CSV only has {len(log_probs_df)} rows"
        )

    scores = []

    for i, aa in enumerate(sequence):
        if aa not in aa_to_col:
            raise ValueError(f"Unknown amino acid '{aa}' at position {i}")

        col = aa_to_col[aa]

        if col not in log_probs_df.columns:
            raise ValueError(f"Column '{col}' not found in log-prob CSV")

        scores.append(float(log_probs_df.iloc[row_offset + i][col]))

    return float(np.mean(scores))


def add_model_scores(df, log_probs_csv, score_col):
    log_probs_df = pd.read_csv(log_probs_csv)

    df[score_col] = df["sequence"].apply(
        lambda seq: sequence_average_log_prob(
            seq,
            log_probs_df,
            row_offset=chain_b_start_row,
        )
    )

    return df


def get_topk_overlap(df, score_col, k):
    target_topk = df.sort_values("target", ascending=False).head(k)
    model_topk = df.sort_values(score_col, ascending=False).head(k)

    target_ids = set(target_topk["row_id"])
    model_ids = set(model_topk["row_id"])
    overlap_ids = target_ids & model_ids

    return {
        "overlap_count": len(overlap_ids),
        "overlap_fraction": len(overlap_ids) / k,
        "target_topk_ids": target_ids,
        "model_topk_ids": model_ids,
        "overlap_ids": overlap_ids,
    }


def get_split_dfs(df):
    split_dfs = {}

    if "set" in df.columns:
        set_col = df["set"].astype(str).str.lower()

        if "validation" in df.columns:
            val_col = df["validation"].astype(bool)

            split_dfs["train"] = df[(set_col == "train") & (~val_col)].copy()
            split_dfs["val"] = df[(set_col == "train") & (val_col)].copy()
            split_dfs["test"] = df[set_col == "test"].copy()

        else:
            split_dfs["train"] = df[set_col == "train"].copy()
            split_dfs["val"] = df[set_col.isin(["val", "valid", "validation"])].copy()
            split_dfs["test"] = df[set_col == "test"].copy()

    elif "split" in df.columns:
        split_col = df["split"].astype(str).str.lower()

        split_dfs["train"] = df[split_col == "train"].copy()
        split_dfs["val"] = df[split_col.isin(["val", "valid", "validation"])].copy()
        split_dfs["test"] = df[split_col == "test"].copy()

    else:
        raise KeyError(
            f"No split column found. Expected 'set' or 'split'. "
            f"Available columns: {list(df.columns)}"
        )

    return split_dfs


df = pd.read_csv(input_csv)
split_dfs = get_split_dfs(df)

summary_rows = []
detail_rows = []

for split_name, split_df in split_dfs.items():
    if len(split_df) == 0:
        print(f"Skipping {split_name}: no rows found")
        continue

    split_df = split_df.reset_index(drop=True)
    split_df["row_id"] = split_df.index

    print(f"\nProcessing split: {split_name}, rows={len(split_df)}")

    for model in models:
        model_name = model["name"]
        score_col = f"{model_name}_log_prob_score"

        print(f"  Scoring model: {model_name}")

        split_df = add_model_scores(
            split_df,
            model["log_probs_csv"],
            score_col,
        )

        pearson = split_df["target"].corr(
            split_df[score_col],
            method="pearson",
        )

        spearman = split_df["target"].corr(
            split_df[score_col],
            method="spearman",
        )

        for k in topk_values:
            if len(split_df) < k:
                print(
                    f"  Skipping Top-{k} for {split_name}/{model_name}: "
                    f"only {len(split_df)} rows"
                )
                continue

            result = get_topk_overlap(split_df, score_col, k)

            summary_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "k": k,
                    "num_samples": len(split_df),
                    "overlap_count": result["overlap_count"],
                    "overlap_fraction": result["overlap_fraction"],
                    "pearson_target_vs_log_prob": pearson,
                    "spearman_target_vs_log_prob": spearman,
                }
            )

            selected_ids = result["target_topk_ids"] | result["model_topk_ids"]

            for _, row in split_df[split_df["row_id"].isin(selected_ids)].iterrows():
                row_id = row["row_id"]

                detail_rows.append(
                    {
                        "split": split_name,
                        "k": k,
                        "model": model_name,
                        "row_id": row_id,
                        "sequence": row["sequence"],
                        "target": row["target"],
                        "model_score": row[score_col],
                        "in_target_topk": row_id in result["target_topk_ids"],
                        "in_model_topk": row_id in result["model_topk_ids"],
                        "overlaps_target_topk": row_id in result["overlap_ids"],
                    }
                )


summary_df = pd.DataFrame(summary_rows)
detail_df = pd.DataFrame(detail_rows)

summary_df.to_csv(output_overlap_csv, index=False)
detail_df.to_csv(output_ranked_csv, index=False)


# -------------------------
# Make one grouped plot per split
# x-axis: models
# bars: Top-10, Top-100, Top-500, Top-1000
# -------------------------

for split_name in summary_df["split"].unique():
    split_plot_df = summary_df[summary_df["split"] == split_name].copy()

    if split_plot_df.empty:
        continue

    model_order = [model["name"] for model in models]
    k_order = topk_values

    x = np.arange(len(model_order))
    width = 0.8 / len(k_order)

    plt.figure(figsize=(14, 6))

    for i, k in enumerate(k_order):
        k_df = split_plot_df[split_plot_df["k"] == k].copy()

        overlap_counts = []
        labels = []

        for model_name in model_order:
            row = k_df[k_df["model"] == model_name]

            if len(row) == 0:
                overlap_counts.append(np.nan)
                labels.append("")
            else:
                overlap = int(row.iloc[0]["overlap_count"])
                pearson = row.iloc[0]["pearson_target_vs_log_prob"]
                spearman = row.iloc[0]["spearman_target_vs_log_prob"]

                overlap_counts.append(overlap)
                labels.append(
                    f"{overlap}/{k}\n"
                    f"p={pearson:.3f}\n"
                    f"s={spearman:.3f}"
                )

        bar_positions = x + (i - (len(k_order) - 1) / 2) * width

        bars = plt.bar(
            bar_positions,
            overlap_counts,
            width,
            label=f"Top-{k}",
        )

        for bar, label in zip(bars, labels):
            if label == "" or np.isnan(bar.get_height()):
                continue

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(k_order) * 0.01,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.ylabel("Overlap count with true Target Top-K")
    plt.title(f"{split_name.upper()} Top-K Retrieval Overlap with Target Ranking")

    plt.xticks(x, model_order, rotation=30, ha="right")
    plt.ylim(0, max(k_order) * 1.2)

    plt.legend(title="Retrieval K")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / f"{split_name}_topk_grouped_bar_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved grouped plot: {plot_path}")


print("\nSummary:")
print(summary_df)

print(f"\nSaved summary CSV to: {output_overlap_csv}")
print(f"Saved detailed CSV to: {output_ranked_csv}")