import argparse
import glob
import os
import signal
import time
from datetime import datetime

_STOP_REQUESTED = False


def _handle_signal(_signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def read_int(path):
    with open(path, "r") as f:
        return int(f.read().strip())


def find_rapl_paths():
    base = "/sys/class/powercap"
    # Prefer package domain if present.
    package_energy = os.path.join(base, "intel-rapl:0", "energy_uj")
    package_max = os.path.join(base, "intel-rapl:0", "max_energy_range_uj")
    if os.path.exists(package_energy) and os.path.exists(package_max):
        return package_energy, package_max
    # Fall back to the first available domain.
    energy_files = glob.glob(os.path.join(base, "intel-rapl:*", "energy_uj"))
    if not energy_files:
        raise RuntimeError("No RAPL energy file found. Are you on an Intel CPU?")
    energy_path = energy_files[0]
    max_path = os.path.join(os.path.dirname(energy_path), "max_energy_range_uj")
    if not os.path.exists(max_path):
        raise RuntimeError(f"Missing max energy range file for {energy_path}")
    return energy_path, max_path


def read_rapl_energy(energy_path):
    """Reads energy in joules from RAPL sysfs."""
    energy_uj = read_int(energy_path)
    return energy_uj / 1_000_000  # convert microjoules to joules


def average_cpu_power(duration_sec=60, sample_interval=1.0, output_dir=None):
    """Computes average CPU package power over a period in seconds.

    If duration_sec is None, runs until interrupted via SIGINT/SIGTERM.
    """
    energy_path, max_path = find_rapl_paths()
    max_energy_j = read_int(max_path) / 1_000_000
    start_energy = read_rapl_energy(energy_path)
    start_time = time.monotonic()

    end_energy = start_energy
    while True:
        if _STOP_REQUESTED:
            break
        if duration_sec is not None and (time.monotonic() - start_time) >= duration_sec:
            break
        time.sleep(sample_interval)
        end_energy = read_rapl_energy(energy_path)

    elapsed = time.monotonic() - start_time
    delta = end_energy - start_energy
    if delta < 0:
        # Counter wrapped.
        delta += max_energy_j
    avg_power = delta / elapsed if elapsed > 0 else 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"start energy is {start_energy}, end energy is {end_energy}")
    print(f"duration sec is {elapsed:.2f}")
    print(f"Average CPU package power over {elapsed:.2f}s: {avg_power:.2f} W")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cpu_power_{timestamp}.txt")
    with open(output_path, "w") as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"start_energy_j: {start_energy}\n")
        f.write(f"end_energy_j: {end_energy}\n")
        f.write(f"duration_sec: {elapsed:.2f}\n")
        f.write(f"avg_power_w: {avg_power}\n")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure average CPU power via Intel RAPL.")
    parser.add_argument("--duration-sec", type=float, default=600.0, help="Duration to measure (seconds).")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Sampling interval (seconds).")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output file.")
    parser.add_argument(
        "--until-interrupt",
        action="store_true",
        help="Run until interrupted (SIGINT/SIGTERM), ignoring duration.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    duration = None if args.until_interrupt else args.duration_sec
    average_cpu_power(duration, sample_interval=args.sample_interval, output_dir=args.output_dir)
