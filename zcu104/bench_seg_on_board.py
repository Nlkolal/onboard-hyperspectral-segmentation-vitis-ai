#!/usr/bin/env python3
import argparse
import time
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


def run_one_image(runner, in_t, out_t, input_scale, data_path, patch):
    # Run DPU segmentation on a single data_X.npy. Returns seconds taken.
    t0 = time.time()

    X = np.load(data_path, mmap_mode='r')  # (H, W, K)
    X = X[:512, :1024, :]
    assert X.shape[0] % patch == 0 and X.shape[1] % patch == 0
    H, W, K = X.shape

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

    t1 = time.time()
    return t1 - t0, (H_valid, W_valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xmodel", type=str, required=True,
        help="Path to compiled .xmodel",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data_X.npy to benchmark on.",
    )
    parser.add_argument(
        "--patch", type=int, default=64,
        help="Patch size used in training (default 64).",
    )
    parser.add_argument(
        "--n_runs", type=int, default=10,
        help="Number of timed runs (default 10).",
    )
    args = parser.parse_args()

    xmodel_path = Path(args.xmodel)
    data_path = Path(args.data)
    patch = args.patch
    n_runs = args.n_runs

    print(f"[info] xmodel : {xmodel_path}")
    print(f"[info] data   : {data_path}")
    print(f"[info] patch  : {patch}")
    print(f"[info] n_runs : {n_runs}")

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

    print("[bench] warmup run...")
    _t_warm, (H_valid, W_valid) = run_one_image(
        runner, in_t, out_t, input_scale, data_path, patch
    )

    times = []
    for i in range(n_runs):
        t, _ = run_one_image(runner, in_t, out_t, input_scale, data_path, patch)
        times.append(t)
        print(f"[bench] run {i+1}/{n_runs}: {t*1000:.2f} ms")

    times = np.array(times)
    avg = times.mean()
    std = times.std()
    min_t = times.min()
    max_t = times.max()

    print("\n[bench] ===== RESULTS =====")
    print(f"[bench] image size (valid region): {H_valid} x {W_valid}")
    print(f"[bench] avg latency : {avg*1000:.2f} ms")
    print(f"[bench] std latency : {std*1000:.2f} ms")
    print(f"[bench] min latency : {min_t*1000:.2f} ms")
    print(f"[bench] max latency : {max_t*1000:.2f} ms")
    print(f"[bench] avg FPS     : {1.0/avg:.2f} images/s")


if __name__ == "__main__":
    main()
