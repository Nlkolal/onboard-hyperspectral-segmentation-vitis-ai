from pathlib import Path
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description="Compute per-band normalization stats on train split and normalize all splits.")
parser.add_argument("--base", type=Path, required=True, help="Folder containing data_*.npy and label_*.npy.")
parser.add_argument("--out", type=Path, required=True, help="Output folder for normalized dataset.")
parser.add_argument("--train", type=int, nargs="+", required=True, help="Train ids.")
parser.add_argument("--val", type=int, nargs="+", required=True, help="Val ids.")
parser.add_argument("--test", type=int, nargs="+", required=True, help="Test ids.")
args = parser.parse_args()

base = args.base
out = args.out
train, val, test = args.train, args.val, args.test

if not base.is_dir():
    sys.exit(f"Base folder does not exist: {base}")

if out.exists():
    sys.exit("the target folder is already created, write a new")
out.mkdir(parents=True, exist_ok=False)

# ---- fit stats on train ----
X = []
for i in train:
    p = base / f"data_{i}.npy"
    if not p.exists():
        sys.exit(f"Missing train file: {p}")
    a = np.load(p).astype(np.float32)
    H, W, B = a.shape
    X.append(a.reshape(-1, B))
X = np.vstack(X)

mu = X.mean(0).astype(np.float32)
sd = (X.std(0) + 1e-6).astype(np.float32)
np.save(out / "mu.npy", mu)
np.save(out / "sd.npy", sd)

def norm_and_save(ids):
    for i in ids:
        x_path = base / f"data_{i}.npy"
        y_path = base / f"label_{i}.npy"
        if not x_path.exists() or not y_path.exists():
            print(f"skip (missing): {i}")
            continue

        a = np.load(x_path).astype(np.float32)
        a = (a - mu) / sd
        np.save(out / f"data_{i}.npy", a)

        y = np.load(y_path)
        np.save(out / f"label_{i}.npy", y)

norm_and_save(train)
norm_and_save(val)
norm_and_save(test)

# sanity check on train after normalization
chk = []
for i in train:
    a = np.load(out / f"data_{i}.npy")
    H, W, B = a.shape
    chk.append(a.reshape(-1, B))
chk = np.vstack(chk)
print("train mean (≈0):", chk.mean(0)[:5])
print("train std  (≈1):", chk.std(0)[:5])
print("done")
