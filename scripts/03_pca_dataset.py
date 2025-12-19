from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import argparse
import sys

parser = argparse.ArgumentParser(description="Fit PCA on train split (normalized data) and transform all splits.")
parser.add_argument("--base", type=Path, required=True, help="Folder containing normalized data_*.npy and label_*.npy.")
parser.add_argument("--out", type=Path, required=True, help="Output folder for PCA-transformed dataset.")
parser.add_argument("--k", type=int, required=True, help="Number of PCA components to keep.")
parser.add_argument("--train", type=int, nargs="+", required=True, help="Train ids.")
parser.add_argument("--val", type=int, nargs="+", required=True, help="Val ids.")
parser.add_argument("--test", type=int, nargs="+", required=True, help="Test ids.")
args = parser.parse_args()

base = args.base
out = args.out
K = args.k
train, val, test = args.train, args.val, args.test

if not base.is_dir():
    sys.exit(f"Base folder does not exist: {base}")

if out.exists():
    sys.exit("the target folder is already created, write a new")
out.mkdir(parents=True, exist_ok=False)

# ---- fit PCA on TRAIN (data already normalized) ----
X = []
for i in train:
    p = base / f"data_{i}.npy"
    if not p.exists():
        sys.exit(f"Missing train file: {p}")
    a = np.load(p).astype(np.float32)
    H, W, B = a.shape
    X.append(a.reshape(-1, B))
X = np.vstack(X)

pca = PCA(n_components=K, whiten=False, random_state=0).fit(X)

# save PCA params
np.save(out / "pca_components.npy", pca.components_.astype(np.float32))
np.save(out / "pca_mean.npy", pca.mean_.astype(np.float32))
np.save(out / "explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))

def transform_and_save(ids):
    for i in ids:
        x_path = base / f"data_{i}.npy"
        y_path = base / f"label_{i}.npy"
        if not x_path.exists() or not y_path.exists():
            print(f"skip (missing): {i}")
            continue

        a = np.load(x_path).astype(np.float32)   # (H,W,B)
        H, W, B = a.shape
        Z = pca.transform(a.reshape(-1, B)).reshape(H, W, K).astype(np.float32)
        np.save(out / f"data_{i}.npy", Z)

        y = np.load(y_path)
        np.save(out / f"label_{i}.npy", y)

transform_and_save(train)
transform_and_save(val)
transform_and_save(test)

print("saved PCA dataset to:", out)
print("explained variance ratio (first 10):", np.round(pca.explained_variance_ratio_[:10], 4))
