#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import xir
import vart


def get_dpu_sub(graph: "xir.Graph"):
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    for cs in children:
        if cs.has_attr("device") and str(cs.get_attr("device")).upper() == "DPU":
            return cs
    raise RuntimeError("No DPU subgraph with device='DPU' found.")


def miou_accuracy(pred, gt, ncls=3):
    """
    pred, gt: 1D numpy arrays of same length (int labels 0..ncls-1).
    Returns: (mIoU, iou_per_class, accuracy).
    """
    cm = np.bincount(ncls * gt + pred, minlength=ncls * ncls).reshape(ncls, ncls)
    tp = np.diag(cm)
    den = cm.sum(1) + cm.sum(0) - tp
    iou = tp / np.maximum(den, 1)
    acc = (pred == gt).mean()
    return float(np.nanmean(iou)), iou, float(acc)


def run_one_image(runner, in_t, out_t, input_scale, data_path, patch, n_classes):
    """Run DPU segmentation on a single data_X.npy. Returns (seg_map, H_valid, W_valid)."""
    X = np.load(data_path, mmap_mode='r')  # (H, W, K)
    H, W, K = X.shape
    print(f"[info]   data: {data_path.name}, shape=({H}, {W}, {K})")

    in_shape = tuple(in_t.dims)
    out_shape = tuple(out_t.dims)

    # layout
    if in_shape[1] == K:
        layout = "NCHW"
        _, _, pin_h, pin_w = in_shape
    elif in_shape[-1] == K:
        layout = "NHWC"
        _, pin_h, pin_w, _ = in_shape
    else:
        raise RuntimeError(
            f"Cannot deduce layout from input dims {in_shape} and channels K={K}."
        )

    assert pin_h == patch and pin_w == patch, \
        f"Model expects {pin_h}x{pin_w} patches, script assumes {patch}."

    H_tiles = H // patch
    W_tiles = W // patch
    H_valid = H_tiles * patch
    W_valid = W_tiles * patch

    if H_valid != H or W_valid != W:
        print(
            f"[warn]   cropping from (H={H}, W={W}) "
            f"to (H_valid={H_valid}, W_valid={W_valid})"
        )

    seg_map = np.zeros((H_valid, W_valid), dtype=np.uint8)

    # tile loop
    for ty in range(H_tiles):
        for tx in range(W_tiles):
            y0 = ty * patch
            x0 = tx * patch

            patch_f = np.array(
                X[y0:y0 + patch, x0:x0 + patch, :],
                dtype=np.float32,
            )  # (P,P,K)

            if layout == "NCHW":
                patch_f = np.transpose(patch_f, (2, 0, 1))  # (K,P,P)

            patch_q = patch_f * input_scale
            patch_q = np.clip(np.round(patch_q), -128, 127).astype(np.int8)

            inp = np.zeros(in_shape, dtype=np.int8)
            out = np.zeros(out_shape, dtype=np.int8)

            if layout == "NCHW":
                inp[0, :, :, :] = patch_q
            else:
                inp[0, :, :, :] = patch_q

            job_id = runner.execute_async([inp], [out])
            runner.wait(job_id)

            if layout == "NCHW":
                logits = out[0, :, :, :]          # (C,P,P)
                pred = np.argmax(logits, axis=0)  # (P,P)
            else:
                logits = out[0, :, :, :]          # (P,P,C)
                pred = np.argmax(logits, axis=-1)

            seg_map[y0:y0 + patch, x0:x0 + patch] = pred.astype(np.uint8)

    return seg_map, H_valid, W_valid


def parse_ids(s: str):
    """Parse comma-separated list like '1,3,16' into [1,3,16]."""
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xmodel",
        type=str,
        required=True,
        help="Path to compiled .xmodel",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset folder containing data_*.npy and label_*.npy",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=64,
        help="Patch size used in training (default 64).",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=3,
        help="Number of segmentation classes (default 3).",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="1,3,16,18,26,28,37,39,41",
        help="Comma-separated image IDs to evaluate, e.g. '1,3,16'.",
    )
    args = parser.parse_args()

    xmodel_path = Path(args.xmodel)
    dataset_dir = Path(args.dataset)
    ids = parse_ids(args.ids)
    patch = args.patch
    n_classes = args.n_classes

    print(f"[info] xmodel : {xmodel_path}")
    print(f"[info] dataset: {dataset_dir}")
    print(f"[info] ids    : {ids}")

    # build runner
    graph = xir.Graph.deserialize(str(xmodel_path))
    dpu_sg = get_dpu_sub(graph)
    runner = vart.Runner.create_runner(dpu_sg, "run")

    in_t = runner.get_input_tensors()[0]
    out_t = runner.get_output_tensors()[0]
    in_shape = tuple(in_t.dims)
    out_shape = tuple(out_t.dims)

    print(f"[runner] input  tensor: shape={in_shape}, dtype={in_t.dtype}")
    print(f"[runner] output tensor: shape={out_shape}, dtype={out_t.dtype}")

    if in_t.has_attr("fix_point"):
        fix_pos = in_t.get_attr("fix_point")
        input_scale = 2 ** fix_pos
    else:
        fix_pos = None
        input_scale = 1.0
    print(f"[quant] input fix_point={fix_pos}, scale={input_scale}")

    # global accumulators
    all_pred = []
    all_gt = []

    for i in ids:
        data_path = dataset_dir / f"data_{i}.npy"
        label_path = dataset_dir / f"label_{i}.npy"

        if not data_path.exists():
            print(f"[warn] data file missing: {data_path}, skipping.")
            continue
        if not label_path.exists():
            print(f"[warn] label file missing: {label_path}, skipping.")
            continue

        print(f"\n[eval] image id {i}")
        seg_map, H_valid, W_valid = run_one_image(
            runner,
            in_t,
            out_t,
            input_scale,
            data_path,
            patch,
            n_classes,
        )

        # load label, crop to valid region
        Y = np.load(label_path).astype(np.int64)  # (H, W)
        H_label, W_label = Y.shape
        if H_label < H_valid or W_label < W_valid:
            raise RuntimeError(
                f"Label for id {i} is too small: ({H_label},{W_label}) vs needed ({H_valid},{W_valid})"
            )
        Y_valid = Y[:H_valid, :W_valid]

        pred_flat = seg_map.ravel()
        gt_flat = Y_valid.ravel()

        all_pred.append(pred_flat)
        all_gt.append(gt_flat)

        miou_i, iou_per_class_i, acc_i = miou_accuracy(pred_flat, gt_flat, ncls=n_classes)
        print(f"[eval] id={i}  acc={acc_i:.4f}  mIoU={miou_i:.4f}  IoU per class={np.round(iou_per_class_i, 4)}")

    if not all_pred:
        print("[error] No valid images evaluated.")
        return

    pred_all = np.concatenate(all_pred)
    gt_all = np.concatenate(all_gt)

    miou, iou_per_class, acc = miou_accuracy(pred_all, gt_all, ncls=n_classes)
    print("\n[eval] === OVERALL ===")
    print(f"[eval] accuracy     : {acc:.4f}")
    print(f"[eval] mIoU         : {miou:.4f}")
    print(f"[eval] IoU per class: {np.round(iou_per_class, 4)}")


if __name__ == "__main__":
    main()
