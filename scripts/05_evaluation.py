# pip install torch torchvision
from pathlib import Path
import argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

from model_architecture.SmallUNet import SmallUNet
from model_architecture.CUNet import CUNet
from model_architecture.CUNetPP import CUNetPP
from model_architecture.CFCN import CFCN
from model_architecture.JustoUNetSimple import JustoUNetSimple


def build_model(name_model: str, in_ch: int, n_classes: int) -> nn.Module:
    name_model = str(name_model).lower()
    if name_model == "cfcn":
        return CFCN(in_ch, n_classes)
    if name_model == "cunet":
        return CUNet(in_ch, n_classes)
    if name_model == "cunetpp":
        return CUNetPP(in_ch, n_classes)
    if name_model == "smallunet":
        return SmallUNet(in_ch, n_classes)
    if name_model == "justounetsimple":
        return JustoUNetSimple(in_ch, n_classes)
    raise ValueError(f"Unknown model name: {name_model}")


def miou_accuracy(pred, gt, ncls=3):
    cm = np.bincount(ncls * gt + pred, minlength=ncls * ncls).reshape(ncls, ncls)
    tp = np.diag(cm)
    den = cm.sum(1) + cm.sum(0) - tp
    iou = tp / np.maximum(den, 1)
    accuracy = (pred == gt).mean()
    return float(np.nanmean(iou)), iou, float(accuracy)


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Evaluate a saved .pt checkpoint on the test set.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_root", type=Path, required=True, help="Root folder that contains the dataset folder")
    parser.add_argument("--test_ids", type=int, nargs="+", default=[1, 3, 16, 18, 26, 28, 37, 39, 41])
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)

    # -----------------
    # load checkpoint
    # -----------------
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    name_model = ckpt.get("model")
    dataset_model = ckpt.get("dataset")
    weights_model = ckpt.get("state_dict")
    epoch_model = ckpt.get("epoch")
    config_model = ckpt.get("config", {})

    PATCH = config_model.get("PATCH") or config_model.get("patch")
    NCLASSES = config_model.get("NCLASSES") or config_model.get("nclasses") or 3
    if PATCH is None:
        raise ValueError("PATCH must be stored in checkpoint['config'].")

    print(f"[ckpt] {args.ckpt}")
    print(f"[ckpt] Model: {name_model}, dataset: {dataset_model}, epoch: {epoch_model}")
    print(f"[ckpt] PATCH={PATCH}, NCLASSES={NCLASSES}")

    # -----------------
    # device
    # -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # -----------------
    # dataset base path
    # -----------------
    base = args.data_root / str(dataset_model)
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    # infer channels K from one file
    X0 = np.load(base / "data_0.npy").astype(np.float32)
    _, _, K0 = X0.shape

    # -----------------
    # rebuild model and load weights
    # -----------------
    model = build_model(name_model, in_ch=K0, n_classes=NCLASSES)
    model.load_state_dict(weights_model)
    model.to(device).eval()
    print(f"[model] Rebuilt {name_model} and loaded weights.")

    # -----------------
    # dataset
    # -----------------
    class ImagePatchesDS(Dataset):
        def __init__(self, ids):
            self.ids = ids
            self.X = [np.load(base / f"data_{i}.npy").astype(np.float32) for i in ids]
            self.Y = [np.load(base / f"label_{i}.npy").astype(np.int64)  for i in ids]
            H, W, K = self.X[0].shape
            print(f"[dataset] {len(ids)} test files, sample shape: (H={H}, W={W}, K={K})")

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            X = self.X[idx]
            Y = self.Y[idx]
            H, W, K = X.shape
            W_max = W // PATCH
            H_max = H // PATCH
            patch_total = W_max * H_max

            x_list = np.empty((PATCH, PATCH, K, patch_total), dtype=np.float32)
            y_list = np.empty((PATCH, PATCH, patch_total), dtype=np.int64)

            for i in range(patch_total):
                x_pix = (i % W_max) * PATCH
                y_pix = (i // W_max) * PATCH
                x_list[:, :, :, i] = X[y_pix:y_pix + PATCH, x_pix:x_pix + PATCH, :]
                y_list[:, :, i] = Y[y_pix:y_pix + PATCH, x_pix:x_pix + PATCH]

            x_tensor = torch.from_numpy(x_list).permute(3, 2, 0, 1).contiguous()
            y_tensor = torch.from_numpy(y_list).permute(2, 0, 1).contiguous()
            return x_tensor, y_tensor

    test_ds = ImagePatchesDS(args.test_ids)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # -----------------
    # evaluation
    # -----------------
    print("\n[eval] Running evaluation on test set...")
    t0 = time.time()

    total_pred, total_gt = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            # xb: (1, Npatch, K, PATCH, PATCH)
            B, Npatch, Kch, Hp, Wp = xb.shape
            xb = xb.view(B * Npatch, Kch, Hp, Wp).to(device, non_blocking=True)
            yb = yb.view(B * Npatch, Hp, Wp).to(device, non_blocking=True)

            logits = model(xb)
            total_pred.append(logits.argmax(1).cpu().numpy().ravel())
            total_gt.append(yb.cpu().numpy().ravel())

    pred = np.concatenate(total_pred)
    gt = np.concatenate(total_gt)
    miou, iou_per_class, accuracy = miou_accuracy(pred, gt, ncls=NCLASSES)
    dt = time.time() - t0

    print("\n[eval] Done.")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  mIoU    : {miou:.4f}")
    print(f"  IoU/cls : {np.round(iou_per_class, 4)}")
    print(f"  Time    : {dt:.2f}s for {len(test_ds)} images")


if __name__ == "__main__":
    main()