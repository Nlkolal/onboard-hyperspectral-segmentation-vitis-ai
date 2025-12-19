#!/usr/bin/env python3
import argparse
import time
from pathlib import Path


def read_int(path: Path):
    with path.open("r") as f:
        return int(f.read().strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hwmon_dir",
        type=str,
        default="/sys/class/hwmon/hwmon0",
        help="Path to hwmon dir (default /sys/class/hwmon/hwmon0).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval in seconds (default 0.5).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Total time to run in seconds. 0 = run forever (default 0).",
    )
    args = parser.parse_args()

    base = Path(args.hwmon_dir)

    power_path = base / "power1_input"
    volt_path = base / "in1_input"
    curr_path = base / "curr1_input"

    if not power_path.exists():
        print(f"[error] {power_path} does not exist")
        return

    print(f"[info] hwmon dir   : {base}")
    print(f"[info] sensor name : {(base / 'name').read_text().strip()}")
    print(f"[info] interval    : {args.interval}s")
    if args.duration > 0:
        print(f"[info] duration    : {args.duration}s\n")
    else:
        print(f"[info] duration    : unlimited (Ctrl+C to stop)\n")

    print("time_s, power_W, voltage_V, current_A")

    t0 = time.time()
    sum_W = 0.0
    n = 0

    while True:
        now = time.time()
        t = now - t0

        # read raw values (µ-units)
        uW = read_int(power_path)      # microwatts
        try:
            uV = read_int(volt_path)   # microvolts
        except FileNotFoundError:
            uV = None
        try:
            uA = read_int(curr_path)   # microamps
        except FileNotFoundError:
            uA = None

        W = uW / 1e6
        V = uV / 1e6 if uV is not None else float("nan")
        A = uA / 1e6 if uA is not None else float("nan")

        sum_W += W
        n += 1

        print(f"{t:.3f}, {W:.4f}, {V:.3f}, {A:.3f}")

        if args.duration > 0 and t >= args.duration:
            break

        time.sleep(args.interval)

    if n > 0:
        avg_W = sum_W / n
        print(f"\n[info] average power over {t:.2f}s: {avg_W:.4f} W")


if __name__ == "__main__":
    main()
