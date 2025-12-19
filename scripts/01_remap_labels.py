from pathlib import Path
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description="Remap labels from {1,2,3} to {0,1,2}.")
parser.add_argument("--base", type=Path, required=True, help="Folder containing label_*.npy files.")
parser.add_argument("--ids", type=int, nargs="+", required=True, help="List of label indices to remap.")
args = parser.parse_args()

base = args.base
ids = args.ids

if not base.is_dir():
    sys.exit(f"Base folder does not exist: {base}")

for i in ids:
    p = base / f"label_{i}.npy"
    if not p.exists():
        print(f"skip (missing): {p}")
        continue

    y = np.load(p)
    y2 = y.copy()
    y2[y == 1] = 0
    y2[y == 2] = 1
    y2[y == 3] = 2
    np.save(p, y2)

print("remapped labels to {0,1,2}")