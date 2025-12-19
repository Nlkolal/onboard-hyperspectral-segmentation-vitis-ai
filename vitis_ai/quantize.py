from pathlib import Path
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_architecture.SmallUNet import SmallUNet
from model_architecture.CUNet import CUNet
from model_architecture.CUNetPP import CUNetPP
from model_architecture.CFCN import CFCN
from model_architecture.JustoUNetSimple import JustoUNetSimple

# Vitis AI quantization API
from pytorch_nndct.apis import torch_quantizer, dump_xmodel



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
    raise ValueError(f"Unknown model name in checkpoint: {name_model}")


# -----------------
# metrics
# -----------------
def miou_accuracy(pred, gt, ncls=3):
    cm = np.bincount(ncls * gt + pred, minlength=ncls * ncls).reshape(ncls, ncls)
    tp = np.diag(cm)
    den = cm.sum(1) + cm.sum(0) - tp
    iou = tp / np.maximum(den, 1)
    accuracy = (pred == gt).mean()
    return float(np.nanmean(iou)), iou, float(accuracy)


# -----------------
# datasets
# -----------------
class ImagePatchesDS(Dataset):
    """Returns all non-overlapping PATCHxPATCH patches for each image."""
    def __init__(self, base: Path, ids, patch: int):
        self.base = Path(base)
        self.ids = list(ids)
        self.patch = int(patch)
        self.X = [np.load(self.base / f"data_{i}.npy").astype(np.float32) for i in self.ids]
        self.Y = [np.load(self.base / f"label_{i}.npy").astype(np.int64) for i in self.ids]
        H, W, K = self.X[0].shape
        print(f"[dataset] {len(self.ids)} files, sample: H={H}, W={W}, K={K}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        P = self.patch

        H, W, K = X.shape
        if H < P or W < P:
            raise ValueError(f"PATCH={P} too large for image {H}x{W}")

        W_max = W // P
        H_max = H // P
        patch_total = W_max * H_max

        x_list = np.empty((P, P, K, patch_total), dtype=np.float32)
        y_list = np.empty((P, P, patch_total), dtype=np.int64)

        for i in range(patch_total):
            x_pix = (i % W_max) * P
            y_pix = (i // W_max) * P
            x_list[:, :, :, i] = X[y_pix:y_pix + P, x_pix:x_pix + P, :]
            y_list[:, :, i] = Y[y_pix:y_pix + P, x_pix:x_pix + P]

        x_tensor = torch.from_numpy(x_list).permute(3, 2, 0, 1).contiguous()  # (Npatch,K,P,P)
        y_tensor = torch.from_numpy(y_list).permute(2, 0, 1).contiguous()     # (Npatch,P,P)
        return x_tensor, y_tensor


class CalibRandomPatchDS(Dataset):
    """Random PATCHxPATCH crops for PTQ calibration."""
    def __init__(self, base: Path, ids, patch: int, length: int = 500, seed: int = 0):
        self.base = Path(base)
        self.ids = list(ids)
        self.patch = int(patch)
        self.length = int(length)
        self.rng = random.Random(seed)
        self.X = [np.load(self.base / f"data_{i}.npy").astype(np.float32) for i in self.ids]

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        X = self.rng.choice(self.X)
        P = self.patch
        H, W, K = X.shape
        if H <= P or W <= P:
            raise ValueError(f"PATCH={P} too large for image {H}x{W}")
        i = self.rng.randrange(H - P)
        j = self.rng.randrange(W - P)
        patch = X[i:i + P, j:j + P, :]
        return torch.from_numpy(patch).permute(2, 0, 1).contiguous()  # (K,P,P)


# -----------------
# evaluation helper
# -----------------
@torch.no_grad()
def evaluate_segmentation(model, test_dl, device, nclasses: int):
    print("\n[eval] Running evaluation on test set...")
    t0 = time.time()
    model.eval()

    total_pred, total_gt = [], []
    for xb, yb in test_dl:
        # xb: (1, Npatch, K, P, P)
        B, Npatch, Kch, Hp, Wp = xb.shape
        xb = xb.view(B * Npatch, Kch, Hp, Wp).to(device, non_blocking=True)
        yb = yb.view(B * Npatch, Hp, Wp).to(device, non_blocking=True)

        logits = model(xb)
        total_pred.append(logits.argmax(1).cpu().numpy().ravel())
        total_gt.append(yb.cpu().numpy().ravel())

    pred = np.concatenate(total_pred)
    gt = np.concatenate(total_gt)
    miou, iou_per_class, acc = miou_accuracy(pred, gt, ncls=nclasses)
    dt = time.time() - t0

    print("[eval] Done.")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  mIoU    : {miou:.4f}")
    print(f"  IoU/cls : {np.round(iou_per_class, 4)}")
    print(f"  Time    : {dt:.2f}s for {len(test_dl.dataset)} images")
    return miou, iou_per_class, acc, dt


# -----------------
# main
# -----------------
def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Eval + (optional) Vitis AI PTQ for segmentation.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_root", type=Path, default=Path("/workspace/dataset"),
                        help="Dataset root in the docker container")
    parser.add_argument("--test_ids", type=int, nargs="+", default=[1, 3, 16, 18, 26, 28, 37, 39, 41])

    parser.add_argument("--quant_mode", type=str, default="float",
                        choices=["float", "calib", "test"],
                        help="'float'=no quant, 'calib'=PTQ calibration, 'test'=use quantized model")
    parser.add_argument("--calib_batches", type=int, default=100)
    parser.add_argument("--calib_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--deploy", action="store_true",
                        help="Export xmodel (only valid with --quant_mode test)")
    parser.add_argument("--out_dir", type=Path, default=Path("quant_out"),
                        help="Where to write quant artifacts (config/onnx/xmodel)")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.deploy and args.quant_mode != "test":
        print("Warning: --deploy requires --quant_mode test. Disabling deploy for this run.")
        args.deploy = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # -----------------
    # load checkpoint
    # -----------------
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    name_model = ckpt.get("model")
    dataset_model = ckpt.get("dataset")
    weights_model = ckpt.get("state_dict")
    config = ckpt.get("config", {})

    PATCH = config.get("PATCH")
    NCLASSES = config.get("NCLASSES")
    train_ids = config.get("train_ids", None)

    if PATCH is None or NCLASSES is None:
        raise ValueError("PATCH and NCLASSES must be stored in checkpoint['config'].")

    print(f"[ckpt] {args.ckpt}")
    print(f"[ckpt] Model={name_model}, dataset={dataset_model}, PATCH={PATCH}, NCLASSES={NCLASSES}")

    base = args.data_root / dataset_model
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    # infer K from data_0.npy
    X0 = np.load(base / "data_0.npy").astype(np.float32)
    _, _, K0 = X0.shape
    print(f"[data] base={base}  K={K0}")

    # rebuild model
    model = build_model(name_model, in_ch=K0, n_classes=NCLASSES)
    model.load_state_dict(weights_model)
    model.to(device)
    model.eval()
    print("[model] Loaded.")

    # dataloader
    test_ds = ImagePatchesDS(base, args.test_ids, patch=PATCH)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # -----------------
    # float eval
    # -----------------
    if args.quant_mode == "float":
        print("\n[mode] float")
        evaluate_segmentation(model, test_dl, device, NCLASSES)
        return

    # dummy input for quantizer trace
    dummy_input = torch.randn(1, K0, PATCH, PATCH, device=device)

    # IMPORTANT: use output_dir so artifacts land somewhere predictable
    print(f"\n[mode] Vitis AI quant_mode={args.quant_mode}")
    quantizer = torch_quantizer(
        args.quant_mode,      # calib or test
        model,
        (dummy_input,),
        device=device,
        output_dir=str(args.out_dir),
    )
    qmodel = quantizer.quant_model.to(device)

    # -----------------
    # calib
    # -----------------
    if args.quant_mode == "calib":
        if train_ids is None:
            raise ValueError("No train_ids in checkpoint['config'] (needed for calibration).")

        calib_len = args.calib_batches * args.calib_batch_size
        calib_ds = CalibRandomPatchDS(base, train_ids, patch=PATCH, length=calib_len, seed=args.seed)
        calib_dl = DataLoader(calib_ds, batch_size=args.calib_batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)

        print(f"[calib] batches={args.calib_batches}  batch_size={args.calib_batch_size}  total_patches={calib_len}")
        qmodel.train()
        with torch.no_grad():
            for i, xb in enumerate(calib_dl, 1):
                xb = xb.to(device, non_blocking=True)
                _ = qmodel(xb)
                if i >= args.calib_batches:
                    break

        print("[calib] quick eval (after calib, before final deploy):")
        qmodel.eval()
        evaluate_segmentation(qmodel, test_dl, device, NCLASSES)

        print("[calib] exporting quant config + onnx ...")
        quantizer.export_quant_config()
        quantizer.export_onnx_model()
        print(f"[calib] done. artifacts in: {args.out_dir}")
        return

    # -----------------
    # test (quantized)
    # -----------------
    qmodel.eval()

    if not args.deploy:
        print("[test] eval quantized model")
        evaluate_segmentation(qmodel, test_dl, device, NCLASSES)

        print("[test] exporting quant config + onnx ...")
        quantizer.export_quant_config()
        quantizer.export_onnx_model()
        print(f"[test] done. artifacts in: {args.out_dir}")
        return

    # -----------------
    # deploy: export xmodel
    # -----------------
    print("[deploy] exporting xmodel (ensure batch=1 forward first)")
    with torch.no_grad():
        _ = qmodel(torch.randn(1, K0, PATCH, PATCH, device=device))

    quantizer.export_quant_config()
    quantizer.export_onnx_model()
    try:
        quantizer.export_xmodel(deploy_check=False)
        print("[deploy] xmodel export done (export_xmodel).")
    except Exception as e:
        print(f"[deploy] export_xmodel failed: {e}")
        print("[deploy] trying dump_xmodel() fallback...")
        dump_dir = args.out_dir / "xmodel_dump"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_xmodel(quantizer, output_dir=str(dump_dir))
        print(f"[deploy] xmodel export done (dump_xmodel -> {dump_dir}).")

    print("[deploy] eval quantized model (after export)")
    evaluate_segmentation(qmodel, test_dl, device, NCLASSES)
    print(f"[deploy] all artifacts in: {args.out_dir}")


if __name__ == "__main__":
    main()