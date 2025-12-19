# pip install torch torchvision
from pathlib import Path
import argparse
import random, time, copy
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from model_architecture.SmallUNet import SmallUNet
from model_architecture.CUNet import CUNet
from model_architecture.CUNetPP import CUNetPP
from model_architecture.CFCN import CFCN
from model_architecture.JustoUNetSimple import JustoUNetSimple


def to_python(obj):
    """Convert numpy arrays/scalars to pure Python so checkpoints load on older NumPy."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj


def build_model(name: str, in_ch: int, n_classes: int) -> nn.Module:
    name = str(name).lower()
    if name == "cfcn":
        return CFCN(in_ch, n_classes)
    if name == "cunet":
        return CUNet(in_ch, n_classes)
    if name == "cunetpp":
        return CUNetPP(in_ch, n_classes)
    if name == "smallunet":
        return SmallUNet(in_ch, n_classes)
    if name == "justounetsimple":
        return JustoUNetSimple(in_ch, n_classes)
    raise ValueError(f"Unknown model_name: {name}")


def miou_accuracy(pred, gt, ncls=3):
    cm = np.bincount(ncls * gt + pred, minlength=ncls * ncls).reshape(ncls, ncls)
    tp = np.diag(cm)
    den = cm.sum(1) + cm.sum(0) - tp
    iou = tp / np.maximum(den, 1)
    accuracy = (pred == gt).mean()
    return float(np.nanmean(iou)), iou, float(accuracy)


def augment_gpu(xb, yb):
    if torch.rand(1) < 0.5:
        xb = xb.flip(-1); yb = yb.flip(-1)
    if torch.rand(1) < 0.5:
        xb = xb.flip(-2); yb = yb.flip(-2)
    k = torch.randint(0, 4, (1,)).item()
    if k:
        xb = torch.rot90(xb, k, dims=(-2, -1))
        yb = torch.rot90(yb, k, dims=(-2, -1))
    return xb, yb


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Train segmentation model on HYPSO npy dataset.")
    parser.add_argument("--model_name", type=str, default="SmallUNet",
                        choices=["SmallUNet", "CUNet", "CUNetPP", "CFCN", "JustoUNetSimple"])
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Root directory that contains the dataset folder (e.g. datasetH1)")
    parser.add_argument("--dataset_name", type=str, default="pca/h1_pca4",
                        help="Relative dataset folder under data_root")
    parser.add_argument("--out_dir", type=Path, default=Path("model_weights"),
                        help="Where to save checkpoints")
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--nclasses", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_tag", type=str, default="Vitis")
    parser.add_argument("--train_ids", type=int, nargs="+")
    parser.add_argument("--val_ids", type=int, nargs="+")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = args.data_root / args.dataset_name
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires a CUDA GPU.")
    print(f"[device] {device}")

    out = args.out_dir / args.model_name
    out.mkdir(parents=True, exist_ok=True)

    # ---- infer channel count from first train file ----
    K = np.load(base / f"data_{args.train_ids[0]}.npy").shape[-1]
    print(f"[data] base={base}  channels(K)={K}")

    # ---- class weights from train distribution ----
    counts = np.zeros(args.nclasses, dtype=np.int64)
    for i in args.train_ids:
        y = np.load(base / f"label_{i}.npy").ravel()
        for c in range(args.nclasses):
            counts[c] += (y == c).sum()
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[loss] class_weights={weights}")

    class PatchDS(Dataset):
        def __init__(self, ids):
            self.X = [np.load(base / f"data_{i}.npy").astype(np.float32) for i in ids]
            self.Y = [np.load(base / f"label_{i}.npy").astype(np.int64) for i in ids]
            H, W, K2 = self.X[0].shape
            print(f"[dataset] {len(ids)} files, sample shape: (H={H}, W={W}, K={K2})")

        def __len__(self):
            return 500

        def __getitem__(self, _):
            idx = random.randrange(len(self.X))
            x, y = self.X[idx], self.Y[idx]
            H, W, K2 = x.shape
            if H <= args.patch or W <= args.patch:
                raise ValueError(f"PATCH={args.patch} too big for image {H}x{W}")
            i = random.randrange(H - args.patch)
            j = random.randrange(W - args.patch)
            xp = x[i:i + args.patch, j:j + args.patch, :]
            yp = y[i:i + args.patch, j:j + args.patch]
            xp = torch.from_numpy(xp).permute(2, 0, 1)  # (K,H,W)
            yp = torch.from_numpy(yp)
            return xp, yp

    train_dl = DataLoader(PatchDS(args.train_ids), batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(PatchDS(args.val_ids), batch_size=args.batch, shuffle=False,
                        num_workers=0, pin_memory=True)

    model = build_model(args.model_name, in_ch=K, n_classes=args.nclasses).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model_name}(in_ch={K}, ncls={args.nclasses}), params={params/1e3:.3f}k")
    print(f"[train] PATCH={args.patch}, BATCH={args.batch}, EPOCHS={args.epochs}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lossfn = nn.CrossEntropyLoss(weight=class_weights)
    lossfn_val = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")

    best = {"miou": -1.0, "epoch": 0, "state": None, "iou_per_class": None}
    loss_train_list, acc_train_list = [], []
    loss_val_list, acc_val_list = [], []

    scaler = GradScaler(device="cuda")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = 0.0
        n_correct = 0
        n_pix = 0

        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb, yb = augment_gpu(xb, yb)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(xb)
                loss = lossfn(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item() * yb.numel()
            n_pix += yb.numel()
            n_correct += (logits.argmax(1) == yb).sum().item()

        loss_train_list.append(run_loss / n_pix)
        acc_train_list.append(n_correct / n_pix)

        # ---- validation ----
        model.eval()
        loss_val = 0.0
        n_pix_val = 0
        total_pred, total_gt = [], []

        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss_val += lossfn_val(logits, yb).item()
                n_pix_val += yb.numel()
                total_pred.append(logits.argmax(1).cpu().numpy().ravel())
                total_gt.append(yb.cpu().numpy().ravel())

        pred = np.concatenate(total_pred)
        gt = np.concatenate(total_gt)
        miou, iou_per_class, acc = miou_accuracy(pred, gt, ncls=args.nclasses)
        loss_val_list.append(loss_val / n_pix_val)
        acc_val_list.append(acc)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}  Acc {acc:.4f}  mIoU {miou:.3f}  IoU {np.round(iou_per_class,3)}  {dt:.1f}s")

        if miou > best["miou"]:
            best.update({
                "miou": miou,
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
                "iou_per_class": iou_per_class.copy(),
            })
            print(f"  [best] new best mIoU {miou:.3f} at epoch {epoch}")

    print("\nTraining complete.")
    print(f"Best mIoU: {best['miou']:.3f} at epoch {best['epoch']}  IoU/cls: {np.round(best['iou_per_class'],3)}")

    ans = input("Will you save this model? (yes/no): ").strip().lower()
    if ans in {"y", "yes"} and best["state"] is not None:
        tag = f"{args.save_tag}_e{best['epoch']}_miou{best['miou']:.3f}_P{args.patch}_K{K}".replace(".", "-")
        ckpt_path = out / f"{args.model_name}_{tag}.pt"

        raw_ckpt = {
            "model": args.model_name,
            "dataset": args.dataset_name,
            "loss_val_list": loss_val_list,
            "accuracy_val_list": acc_val_list,
            "loss_train_list": loss_train_list,
            "accuracy_train_list": acc_train_list,
            "state_dict": best["state"],  # tensors
            "epoch": best["epoch"],
            "miou": best["miou"],
            "iou_per_class": best["iou_per_class"],  # numpy -> to_python()
            "config": {
                "PATCH": args.patch,
                "BATCH": args.batch,
                "EPOCHS": args.epochs,
                "NCLASSES": args.nclasses,
                "train_ids": args.train_ids,
                "val_ids": args.val_ids,
                "seed": args.seed,
            },
        }

        torch.save(to_python(raw_ckpt), ckpt_path)
        print(f"[save] saved model to: {ckpt_path}")
    else:
        print("[save] skipped saving model.")


if __name__ == "__main__":
    main()
