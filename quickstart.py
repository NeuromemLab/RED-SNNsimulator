from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
PROFILES = {
    "smoke": {
        "max_samples": 4096,
        "batch_size": 1024,
        "mc_trials": 1,
        "weight_error_samples": 4,
    },
    "standard": {
        "max_samples": 100000,
        "batch_size": 2048,
        "mc_trials": 2,
        "weight_error_samples": 16,
    },
    "full": {
        "max_samples": 0,
        "batch_size": 2048,
        "mc_trials": 3,
        "weight_error_samples": 32,
    },
}


def existing_file(text: str, label: str) -> Path:
    expanded = os.path.abspath(os.path.expanduser(text))
    if os.name == "nt" and len(expanded) >= 260 and not expanded.startswith("\\\\?\\"):
        expanded = f"\\\\?\\{expanded}"
    path = Path(expanded)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def windows_long_path(path: Path) -> str:
    text = str(path)
    if os.name == "nt" and len(text) >= 260 and not text.startswith("\\\\?\\"):
        return f"\\\\?\\{text}"
    return text


def default_output_dir(profile: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / f"tsnf_rram_{profile}_{timestamp}"


def build_command(args: argparse.Namespace) -> list[str]:
    profile = PROFILES[args.profile]
    checkpoint = existing_file(args.checkpoint, "checkpoint")
    dataset = existing_file(args.dataset, "dataset")
    output = Path(args.output).expanduser().resolve() if args.output else default_output_dir(args.profile)
    cache = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else ROOT / ".cache" / "dataset"
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    command = [
        sys.executable,
        str(ROOT / "evaluate_tsnf_rram.py"),
        "--checkpoint",
        windows_long_path(checkpoint),
        "--data-dir",
        windows_long_path(dataset),
        "--rram-csv",
        str(ROOT / "data" / "read_res_arr_28nm.csv"),
        "--conv1-q",
        str(ROOT / "data" / "ptq_conv1_int3_per_layer_code_matrix.csv"),
        "--cache-dir",
        str(cache),
        "--out-dir",
        str(output),
        "--device",
        device,
        "--max-samples",
        str(profile["max_samples"]),
        "--batch-size",
        str(profile["batch_size"]),
        "--mc-trials",
        str(profile["mc_trials"]),
        "--weight-error-samples",
        str(profile["weight_error_samples"]),
    ]
    if args.trust_checkpoint:
        command.append("--trust-checkpoint")
    if args.hash_inputs:
        command.append("--hash-inputs")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TSNF evaluation with Int3 and static empirical RRAM mapping. "
            "The smoke profile is the recommended first run."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("TSNF_CHECKPOINT", ""),
        required="TSNF_CHECKPOINT" not in os.environ,
        help="Trained TSNF checkpoint, or set TSNF_CHECKPOINT.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TSNF_DATASET", ""),
        required="TSNF_DATASET" not in os.environ,
        help="NPZ archive containing spikes and labels, or set TSNF_DATASET.",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0.")
    parser.add_argument("--output", default="", help="Optional result directory.")
    parser.add_argument("--cache-dir", default="", help="Optional reusable dataset-cache directory.")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow legacy pickle loading only when the checkpoint source is trusted.",
    )
    parser.add_argument("--hash-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print(f"Profile: {args.profile}", flush=True)
    print(f"Device: {command[command.index('--device') + 1]}", flush=True)
    print(f"Results: {command[command.index('--out-dir') + 1]}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
