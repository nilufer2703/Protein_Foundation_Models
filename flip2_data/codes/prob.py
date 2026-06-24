#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")
NUM_WORKERS = 4

LOG_FOLDER = Path("/dapustor/nilufer/ProteinMPNN/INFERENCE/logging")


parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", required=True)
parser.add_argument("--output_folder", required=True)
args = parser.parse_args()

INPUT_FOLDER = Path(args.input_folder)
OUTPUT_FOLDER = Path(args.output_folder)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_FOLDER.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_FOLDER / "extract_log_probs_single.log"
parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", required=True)
parser.add_argument("--output_folder", required=True)

args = parser.parse_args()

INPUT_FOLDER = Path(args.input_folder)
OUTPUT_FOLDER = Path(args.output_folder)

def extract_log_probs(npz_file):
   
    try:
        data = np.load(npz_file)

        if "log_probs" not in data:
            return f"[SKIP] {npz_file.name}: no 'log_probs' key"

        log_probs = data["log_probs"]

        if log_probs.ndim < 2:
            return f"[ERROR] {npz_file.name}: invalid log_probs shape {log_probs.shape}"

        log_probs = log_probs[0]

        
        if log_probs.ndim != 2:
            return f"[ERROR] {npz_file.name}: unexpected shape after indexing {log_probs.shape}"

        if log_probs.shape[1] != len(AMINO_ACIDS):
            return (
                f"[ERROR] {npz_file.name}: expected {len(AMINO_ACIDS)} AA columns, "
                f"got {log_probs.shape[1]}"
            )

        df = pd.DataFrame(log_probs, columns=AMINO_ACIDS)

        out_csv = OUTPUT_FOLDER / f"{npz_file.stem}.csv"
        df.to_csv(out_csv, index=False)

        return f"[SAVED] {out_csv.name}  shape={log_probs.shape}"

    except Exception as e:
        return f"[ERROR] {npz_file.name}: {e}"


def main():
    npz_files = sorted(INPUT_FOLDER.glob("*.npz"))

    print(f"Found {len(npz_files)} NPZ files")
    print(f"Using {NUM_WORKERS} CPU cores")

    results = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(extract_log_probs, f): f for f in npz_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting log_probs"):
            results.append(future.result())

    # Write log
    with open(LOG_FILE, "w") as f:
        for line in results:
            f.write(line + "\n")

    print("\nDone")
    print(f"Log saved to: {LOG_FILE}")
    print(f"CSV output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
