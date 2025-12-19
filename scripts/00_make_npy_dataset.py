from pathlib import Path
from hypso import Hypso1, Hypso2
import numpy as np
import sys
import argparse


parser = argparse.ArgumentParser(description="Convert HYPSO .nc + .dat folders into paired .npy cubes and labels.")
parser.add_argument("--dir_base", type=Path, required=True
                    help="Base directory containing per-scene folders with .nc and .dat files.")
parser.add_argument("--dir_target", type=Path, required=True
                    help="Output directory where data_i.npy and label_i.npy are written.")
parser.add_argument("--mission", choices=["H1", "H2"], default="H1",
                    help="Which HYPSO mission reader to use (H1 or H2).")
parser.add_argument("--verbose", action="store_true",
                    help="Print extra debug information.")
args = parser.parse_args()

dir_base = args.dir_base
dir_target = args.dir_target


if dir_target.is_dir():
    sys.exit("the target folder for this dataset is already created, write a new")
else:
    dir_target.mkdir(parents=True, exist_ok=False)
if not dir_base.is_dir():
    sys.exit(f"dir_base does not exist or is not a directory: {dir_base}")


def load_hyperspectral_dataH2(Load_NC_Path: Path, verbose: bool = False):
    satobj_h2 = Hypso2(path=Load_NC_Path, verbose=verbose)
    satobj_h2.generate_l1b_cube()
    data_cube = satobj_h2.l1b_cube.to_numpy()
    data_cube = processing_drop_bandsH2(data_cube)
    return data_cube.astype(np.float32, copy=False)


def load_hyperspectral_dataH1(Load_NC_Path: Path, verbose: bool = False):
    satobj_h1 = Hypso1(path=Load_NC_Path, verbose=verbose)
    satobj_h1.generate_l1b_cube()
    data_cube = satobj_h1.l1b_cube.to_numpy()
    if verbose:
        print("dtype:", data_cube.dtype)
        print("min/max:", float(np.min(data_cube)), float(np.max(data_cube)))
    data_cube = processing_drop_bandsH1(data_cube)
    return data_cube.astype(np.float32, copy=False)


def processing_drop_bandsH2(data_cube):
    drop = [0, 1, 2, 3, 4, 5, 6, 7, 119, 118]
    data_cube = np.delete(data_cube, drop, axis=-1)
    return data_cube


def processing_drop_bandsH1(data_cube):
    drop = [0, 1, 2, 3, 4, 5, 119, 118, 117]
    data_cube = np.delete(data_cube, drop, axis=-1)
    return data_cube


def processing_keep_every_n(data_cube, n):
    drop = [0, 1, 2, 3, 4, 5, 119, 118, 117]
    data_cube = np.delete(data_cube, drop, axis=-1)
    return data_cube[..., ::n]


def load_labels(dat_path: Path, H: int, W: int) -> np.ndarray:
    return np.fromfile(dat_path, dtype=np.uint8).reshape(H, W)


def get_first_sorted(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern))
    return files[0] if files else None


i = 0
for p in sorted(dir_base.iterdir()):
    H, W, B = 0, 0, 0
    if p.is_dir():
        if args.verbose:
            print("folder:", p.name)

        nc_file = get_first_sorted(p, "*.nc")
        dat_file = get_first_sorted(p, "*.dat")

        if nc_file is None or dat_file is None:
            if args.verbose:
                print(f"  skipped (missing .nc or .dat): {p}")
            continue

        path_nc = dir_target / f"data_{i}.npy"
        path_dat = dir_target / f"label_{i}.npy"

        if args.mission == "H2":
            data_cube = load_hyperspectral_dataH2(nc_file, verbose=args.verbose)
        else:
            data_cube = load_hyperspectral_dataH1(nc_file, verbose=args.verbose)

        H, W, B = data_cube.shape
        np.save(path_nc, data_cube)

        labels = load_labels(dat_file, H, W)
        np.save(path_dat, labels)

        i += 1
