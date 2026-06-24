import re
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

parser = argparse.ArgumentParser()

parser.add_argument("--log_file", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--model_name", required=True)

args = parser.parse_args()

LOG_FILE = Path(args.log_file)

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = args.model_name
def parse_metrics(line):
    metrics = {}

    for part in line.strip().split("|"):
        part = part.strip()
        if "=" not in part:
            continue

        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value

    return metrics


def parse_log_file(log_file):
    train_rows = []
    val_rows = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            train_match = re.search(r"Train epoch (\d+)", line)
            val_match = re.search(r"Validation epoch (\d+)", line)

            if train_match:
                metrics = parse_metrics(line)
                metrics["epoch"] = int(train_match.group(1))
                train_rows.append(metrics)

            elif val_match:
                metrics = parse_metrics(line)
                metrics["epoch"] = int(val_match.group(1))
                val_rows.append(metrics)

    train_df = pd.DataFrame(train_rows).sort_values("epoch")
    val_df = pd.DataFrame(val_rows).sort_values("epoch")

    return train_df, val_df


def plot_metrics(
    df,
    x_col,
    y_cols,
    labels,
    title,
    ylabel,
    save_path,
    y_min=None,
    y_max=None,
    y_tick_interval=None,
):
    plt.figure(figsize=(9, 5))

    plotted = False

    for y_col, label in zip(y_cols, labels):
        if y_col in df.columns:
            plt.plot(
                df[x_col],
                df[y_col],
                marker="o",
                linewidth=2,
                markersize=4,
                label=label,
            )
            plotted = True

    if not plotted:
        print(f"Skipping {title}: no matching columns found.")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    if y_tick_interval is not None:
        plt.gca().yaxis.set_major_locator(MultipleLocator(y_tick_interval))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved plot: {save_path}")


train_df, val_df = parse_log_file(LOG_FILE)

train_csv = OUT_DIR / "parsed_train_metrics.csv"
val_csv = OUT_DIR / "parsed_val_metrics.csv"

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print(f"Parsed train metrics: {len(train_df)} epochs")
print(f"Parsed val metrics: {len(val_df)} epochs")
print(f"Saved parsed train CSV: {train_csv}")
print(f"Saved parsed val CSV: {val_csv}")


# -------------------------
# Axis settings
# -------------------------

# Accuracy plots: start at 0.3, end at 1.0, tick every 0.05
ACC_Y_MIN = 0.3
ACC_Y_MAX = 1.0
ACC_TICK = 0.05

# Loss plot: tick every 0.01
# Leave y_min/y_max as None so matplotlib chooses range automatically.
LOSS_Y_MIN = None
LOSS_Y_MAX = None
LOSS_TICK = 0.01


# -------------------------
# Plot 1: Mask-SPPO Loss
# -------------------------

if "val_sppo_loss" in val_df.columns:
    loss_df = train_df.merge(
        val_df[["epoch", "val_sppo_loss"]],
        on="epoch",
        how="outer",
    )
else:
    loss_df = train_df.copy()

plot_metrics(
    df=loss_df,
    x_col="epoch",
    y_cols=["sppo_loss", "val_sppo_loss"],
    labels=["Train Mask-SPPO loss", "Validation Mask-SPPO loss"],
    title="Train vs Validation Mask-SPPO Loss",
    ylabel="Mask-SPPO Loss",
    save_path=OUT_DIR / "mask_sppo_loss_across_epochs.png",
    y_min=LOSS_Y_MIN,
    y_max=LOSS_Y_MAX,
    y_tick_interval=LOSS_TICK,
)


# -------------------------
# Plot 2: Train Full Accuracy
# -------------------------

plot_metrics(
    df=train_df,
    x_col="epoch",
    y_cols=["policy_full_acc", "ref_full_acc"],
    labels=[f"{MODEL_NAME} full acc", "Reference full acc"],
    title="Train Full Accuracy",
    ylabel="Accuracy",
    save_path=OUT_DIR / "train_full_accuracy.png",
    y_min=ACC_Y_MIN,
    y_max=ACC_Y_MAX,
    y_tick_interval=ACC_TICK,
)


# -------------------------
# Plot 3: Validation Full Accuracy
# -------------------------

plot_metrics(
    df=val_df,
    x_col="epoch",
    y_cols=["val_policy_full_acc", "val_ref_full_acc"],
    labels=[f"{MODEL_NAME} full acc", "Reference full acc"],
    title="Validation Full Accuracy",
    ylabel="Accuracy",
    save_path=OUT_DIR / "validation_full_accuracy.png",
    y_min=ACC_Y_MIN,
    y_max=ACC_Y_MAX,
    y_tick_interval=ACC_TICK,
)


# -------------------------
# Plot 4: Train Critical Accuracy
# -------------------------

plot_metrics(
    df=train_df,
    x_col="epoch",
    y_cols=["policy_critical_acc", "ref_critical_acc"],
    labels=[f"{MODEL_NAME} critical acc", "Reference critical acc"],
    title="Train Critical Accuracy",
    ylabel="Accuracy",
    save_path=OUT_DIR / "train_critical_accuracy.png",
    y_min=ACC_Y_MIN,
    y_max=ACC_Y_MAX,
    y_tick_interval=ACC_TICK,
)


# -------------------------
# Plot 5: Validation Critical Accuracy
# -------------------------

plot_metrics(
    df=val_df,
    x_col="epoch",
    y_cols=["val_policy_critical_acc", "val_ref_critical_acc"],
    labels=[f"{MODEL_NAME} critical acc", "Reference critical acc"],
    title="Validation Critical Accuracy",
    ylabel="Accuracy",
    save_path=OUT_DIR / "validation_critical_accuracy.png",
    y_min=ACC_Y_MIN,
    y_max=ACC_Y_MAX,
    y_tick_interval=ACC_TICK,
)

# -------------------------
# Plot 6: Train Region Accuracy
# -------------------------

plot_metrics(
    df=train_df,
    x_col="epoch",
    y_cols=["policy_region_acc", "ref_region_acc"],
    labels=[f"{MODEL_NAME} region acc", "Reference region acc"],
    title="Train Region Accuracy",
    ylabel="Accuracy",
    save_path=OUT_DIR / "train_region_accuracy.png",
    y_min=ACC_Y_MIN,
    y_max=ACC_Y_MAX,
    y_tick_interval=ACC_TICK,
)


# -------------------------
# Plot 7: Validation Region Accuracy
# -------------------------

plot_metrics(
    df=val_df,
    x_col="epoch",
    y_cols=["val_policy_region_acc", "val_ref_region_acc"],
    labels=[f"{MODEL_NAME} region acc", "Reference region acc"],
    title="Validation Region Accuracy",
    ylabel="Accuracy",
    save_path=OUT_DIR / "validation_region_accuracy.png",
    y_min=ACC_Y_MIN,
    y_max=ACC_Y_MAX,
    y_tick_interval=ACC_TICK,
)