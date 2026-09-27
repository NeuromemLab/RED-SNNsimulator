from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset, Subset

from tsnf_rram_eval_utils import (
    SpikesLabelsMemmapDataset,
    best_threshold_from_roc,
    prepare_npy_cache,
    quantize_vec_signed_symmetric,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RRAM_CSV = ROOT / "data" / "read_res_arr_28nm.csv"
DEFAULT_CONV1_Q = ROOT / "data" / "ptq_conv1_int3_per_layer_code_matrix.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "rram_device_nonideal_accuracy"


@dataclass(frozen=True)
class LayerQuantization:
    name: str
    q: np.ndarray
    deq: np.ndarray
    scale: float
    clip: float
    mse: float
    sat_frac: float
    zero_frac: float
    qmin: int
    qmax: int


@dataclass(frozen=True)
class EmpiricalRRAMDistribution:
    resistance_ohm_lrs: np.ndarray
    resistance_ohm_hrs: np.ndarray
    conductance_s_lrs: np.ndarray
    conductance_s_hrs: np.ndarray
    threshold_ohm: float
    delta_g_ref_s: float
    raw_count: int
    discarded_count: int

    def summary(self) -> dict[str, float | int]:
        def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
            r_kohm = values / 1e3
            return {
                f"{prefix}_count": int(values.size),
                f"{prefix}_r_median_kohm": float(np.median(r_kohm)),
                f"{prefix}_r_mean_kohm": float(np.mean(r_kohm)),
                f"{prefix}_r_std_kohm": float(np.std(r_kohm)),
                f"{prefix}_r_p05_kohm": float(np.percentile(r_kohm, 5)),
                f"{prefix}_r_p95_kohm": float(np.percentile(r_kohm, 95)),
                f"{prefix}_g_median_uS": float(np.median(1.0 / values) * 1e6),
                f"{prefix}_g_mean_uS": float(np.mean(1.0 / values) * 1e6),
            }

        out: dict[str, float | int] = {
            "raw_count": self.raw_count,
            "discarded_nonpositive_or_nonfinite_count": self.discarded_count,
            "threshold_kohm": float(self.threshold_ohm / 1e3),
            "delta_g_ref_uS": float(self.delta_g_ref_s * 1e6),
        }
        out.update(stats(self.resistance_ohm_lrs, "lrs"))
        out.update(stats(self.resistance_ohm_hrs, "hrs"))
        return out


def make_dataset(base: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(base):
        return base
    return Subset(base, list(range(max_samples)))


def load_empirical_rram_distribution(path: Path, unit: str = "kohm") -> EmpiricalRRAMDistribution:
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    raw_count = int(arr.size)
    values = arr.reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if unit.lower() in {"kohm", "k"}:
        resistance_ohm = values * 1e3
    elif unit.lower() in {"ohm", "r"}:
        resistance_ohm = values
    else:
        raise ValueError(f"unknown resistance unit: {unit}")

    log_r = np.log10(resistance_ohm)
    centers = np.array([np.percentile(log_r, 20), np.percentile(log_r, 80)], dtype=np.float64)
    for _ in range(100):
        dist = np.abs(log_r[:, None] - centers[None, :])
        labels = np.argmin(dist, axis=1)
        new_centers = np.array(
            [
                log_r[labels == idx].mean() if np.any(labels == idx) else centers[idx]
                for idx in range(2)
            ],
            dtype=np.float64,
        )
        if np.allclose(new_centers, centers, rtol=0, atol=1e-8):
            break
        centers = new_centers

    lrs_label = int(np.argmin(centers))
    hrs_label = 1 - lrs_label
    lrs = resistance_ohm[labels == lrs_label]
    hrs = resistance_ohm[labels == hrs_label]
    if lrs.size == 0 or hrs.size == 0:
        raise ValueError("failed to split the measured resistance array into LRS/HRS clusters")

    threshold_ohm = float(10 ** (0.5 * (centers[lrs_label] + centers[hrs_label])))
    g_lrs = 1.0 / lrs
    g_hrs = 1.0 / hrs
    delta_g_ref = float(np.median(g_lrs) - np.median(g_hrs))
    if delta_g_ref <= 0:
        raise ValueError("invalid empirical conductance split: median LRS conductance <= HRS")
    return EmpiricalRRAMDistribution(
        resistance_ohm_lrs=lrs.astype(np.float32),
        resistance_ohm_hrs=hrs.astype(np.float32),
        conductance_s_lrs=g_lrs.astype(np.float32),
        conductance_s_hrs=g_hrs.astype(np.float32),
        threshold_ohm=threshold_ohm,
        delta_g_ref_s=delta_g_ref,
        raw_count=raw_count,
        discarded_count=raw_count - int(values.size),
    )


def plot_rram_distribution(dist: EmpiricalRRAMDistribution, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    bins_r = np.linspace(
        min(dist.resistance_ohm_lrs.min(), dist.resistance_ohm_hrs.min()) / 1e3,
        max(dist.resistance_ohm_lrs.max(), dist.resistance_ohm_hrs.max()) / 1e3,
        54,
    )
    axes[0].hist(dist.resistance_ohm_lrs / 1e3, bins=bins_r, alpha=0.75, color="#d55e00", label="LRS")
    axes[0].hist(dist.resistance_ohm_hrs / 1e3, bins=bins_r, alpha=0.75, color="#0072b2", label="HRS")
    axes[0].axvline(dist.threshold_ohm / 1e3, color="black", linestyle="--", linewidth=1.2, label="cluster split")
    axes[0].set_xlabel("Measured 1T1R read resistance (kOhm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("28nm measured resistance states")
    axes[0].legend(frameon=False)

    bins_g = np.linspace(
        min(dist.conductance_s_lrs.min(), dist.conductance_s_hrs.min()) * 1e6,
        max(dist.conductance_s_lrs.max(), dist.conductance_s_hrs.max()) * 1e6,
        54,
    )
    axes[1].hist(dist.conductance_s_lrs * 1e6, bins=bins_g, alpha=0.75, color="#d55e00", label="LRS")
    axes[1].hist(dist.conductance_s_hrs * 1e6, bins=bins_g, alpha=0.75, color="#0072b2", label="HRS")
    axes[1].set_xlabel(r"Conductance ($\mu$S)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Conductance used by device sampler")
    axes[1].legend(frameon=False)
    fig.savefig(out_dir / "measured_rram_state_distribution.png", dpi=240)
    plt.close(fig)


def quantize_layer_per_tensor(weight: np.ndarray, bits: int) -> LayerQuantization:
    flat = weight.reshape(-1).astype(np.float32, copy=False)
    deq_flat, stats = quantize_vec_signed_symmetric(flat, bits=bits, method="mse")
    q_flat = np.round(deq_flat / float(stats["scale"])).astype(np.int32)
    q = q_flat.reshape(weight.shape).astype(np.int16)
    deq = deq_flat.reshape(weight.shape).astype(np.float32)
    return LayerQuantization(
        name="",
        q=q,
        deq=deq,
        scale=float(stats["scale"]),
        clip=float(stats["clip"]),
        mse=float(stats["mse"]),
        sat_frac=float(stats["sat_frac"]),
        zero_frac=float(np.mean(q == 0)),
        qmin=int(q.min()),
        qmax=int(q.max()),
    )


def load_quantized_weights(
    checkpoint_path: Path,
    conv1_q_path: Path,
    bits: int,
    trust_checkpoint: bool = False,
) -> tuple[dict[str, torch.Tensor], LayerQuantization, LayerQuantization]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not conv1_q_path.is_file():
        raise FileNotFoundError(f"conv1 code matrix not found: {conv1_q_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        if not trust_checkpoint:
            raise ValueError(
                "this PyTorch version does not support safe weights-only loading; "
                "use --trust-checkpoint only for a checkpoint from a trusted source"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError) as error:
        if not trust_checkpoint:
            raise ValueError(
                "checkpoint contains objects rejected by safe weights-only loading; "
                "use --trust-checkpoint only after verifying its source"
            ) from error
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "net" not in checkpoint:
        raise ValueError("checkpoint must be a mapping containing a 'net' state dictionary")
    state = checkpoint["net"]
    required_keys = {"conv1.weight", "conv2.weight", "Neuron1.w", "Neuron2.w"}
    missing_keys = sorted(required_keys.difference(state))
    if missing_keys:
        raise KeyError(f"checkpoint state is missing required keys: {missing_keys}")
    if tuple(state["conv1.weight"].shape) != (64, 2, 3, 3):
        raise ValueError(f"unexpected conv1 shape: {tuple(state['conv1.weight'].shape)}")
    if tuple(state["conv2.weight"].shape) != (1, 64, 5, 5):
        raise ValueError(f"unexpected conv2 shape: {tuple(state['conv2.weight'].shape)}")
    conv1_weight = state["conv1.weight"].detach().cpu().float().numpy()
    conv2_weight = state["conv2.weight"].detach().cpu().float().numpy()

    conv1_from_checkpoint = quantize_layer_per_tensor(conv1_weight, bits=bits)
    conv1_q_matrix = np.loadtxt(conv1_q_path, delimiter=",", dtype=np.int16)
    conv1_q = conv1_q_matrix.T.reshape(conv1_weight.shape).astype(np.int16)
    if conv1_q.shape != conv1_from_checkpoint.q.shape:
        raise ValueError(f"conv1 q shape mismatch: {conv1_q.shape} vs {conv1_from_checkpoint.q.shape}")
    if not np.array_equal(conv1_q, conv1_from_checkpoint.q):
        diff = int(np.count_nonzero(conv1_q != conv1_from_checkpoint.q))
        raise ValueError(
            f"saved conv1 code matrix differs from checkpoint quantization in {diff} entries"
        )
    conv1_deq = conv1_q.astype(np.float32) * conv1_from_checkpoint.scale
    conv1 = LayerQuantization(
        name="conv1.weight",
        q=conv1_q,
        deq=conv1_deq,
        scale=conv1_from_checkpoint.scale,
        clip=conv1_from_checkpoint.clip,
        mse=float(np.mean((conv1_deq.reshape(-1) - conv1_weight.reshape(-1)) ** 2)),
        sat_frac=conv1_from_checkpoint.sat_frac,
        zero_frac=float(np.mean(conv1_q == 0)),
        qmin=int(conv1_q.min()),
        qmax=int(conv1_q.max()),
    )

    conv2 = quantize_layer_per_tensor(conv2_weight, bits=bits)
    conv2 = LayerQuantization(
        name="conv2.weight",
        q=conv2.q,
        deq=conv2.deq,
        scale=conv2.scale,
        clip=conv2.clip,
        mse=conv2.mse,
        sat_frac=conv2.sat_frac,
        zero_frac=conv2.zero_frac,
        qmin=conv2.qmin,
        qmax=conv2.qmax,
    )
    return state, conv1, conv2


def layer_quant_summary(item: LayerQuantization) -> dict[str, float | int | str | list[int]]:
    values, counts = np.unique(item.q, return_counts=True)
    return {
        "name": item.name,
        "shape": list(item.q.shape),
        "scale": item.scale,
        "clip": item.clip,
        "mse": item.mse,
        "sat_frac": item.sat_frac,
        "zero_frac": item.zero_frac,
        "qmin": item.qmin,
        "qmax": item.qmax,
        "code_histogram": {str(int(v)): int(c) for v, c in zip(values, counts)},
    }


class RRAMWeightSampler:
    def __init__(
        self,
        q_np: np.ndarray,
        scale: float,
        dist: EmpiricalRRAMDistribution,
        device: torch.device,
    ) -> None:
        self.q = torch.as_tensor(q_np, dtype=torch.int16, device=device)
        self.scale = float(scale)
        self.device = device
        self.lrs_values = torch.as_tensor(dist.conductance_s_lrs, dtype=torch.float32, device=device)
        self.hrs_values = torch.as_tensor(dist.conductance_s_hrs, dtype=torch.float32, device=device)
        self.delta_g_ref = float(dist.delta_g_ref_s)
        self.static_weight: torch.Tensor | None = None

    def begin_trial(self) -> None:
        self.static_weight = self.sample_weight()

    def weight_for_timestep(self) -> torch.Tensor:
        if self.static_weight is None:
            self.static_weight = self.sample_weight()
        return self.static_weight

    def sample_weight(self) -> torch.Tensor:
        q_eff = torch.zeros(self.q.shape, dtype=torch.float32, device=self.device)
        for bit in (1, 2):
            gp = self._sample_branch_conductance(bit, "p")
            gn = self._sample_branch_conductance(bit, "n")
            q_eff = q_eff + float(bit) * (gp - gn) / self.delta_g_ref
        return q_eff * self.scale

    def _ideal_lrs_state(self, bit: int, branch: str) -> torch.Tensor:
        magnitude = torch.abs(self.q)
        active = torch.bitwise_and(magnitude, bit) != 0
        if branch == "p":
            return active & (self.q > 0)
        if branch == "n":
            return active & (self.q < 0)
        raise ValueError(f"unknown branch: {branch}")

    def _sample_branch_conductance(self, bit: int, branch: str) -> torch.Tensor:
        is_lrs = self._ideal_lrs_state(bit, branch)
        lrs_idx = torch.randint(0, self.lrs_values.numel(), self.q.shape, device=self.device)
        hrs_idx = torch.randint(0, self.hrs_values.numel(), self.q.shape, device=self.device)
        lrs = self.lrs_values[lrs_idx]
        hrs = self.hrs_values[hrs_idx]
        return torch.where(is_lrs, lrs, hrs)


class RRAMMappedPSIFNet(torch.nn.Module):
    def __init__(
        self,
        checkpoint_state: dict[str, torch.Tensor],
        conv1_sampler: RRAMWeightSampler | None,
        conv2_sampler: RRAMWeightSampler | None,
        conv1_quant: LayerQuantization | None,
        conv2_quant: LayerQuantization | None,
        device: torch.device,
        mode: str,
    ) -> None:
        super().__init__()
        self.device_obj = device
        self.mode = mode
        self.conv1_sampler = conv1_sampler
        self.conv2_sampler = conv2_sampler
        self.conv1_fp32 = checkpoint_state["conv1.weight"].detach().to(device=device, dtype=torch.float32)
        self.conv2_fp32 = checkpoint_state["conv2.weight"].detach().to(device=device, dtype=torch.float32)
        self.conv1_ideal = (
            torch.as_tensor(conv1_quant.deq, dtype=torch.float32, device=device) if conv1_quant is not None else None
        )
        self.conv2_ideal = (
            torch.as_tensor(conv2_quant.deq, dtype=torch.float32, device=device) if conv2_quant is not None else None
        )
        self.theta1 = self._theta(checkpoint_state["Neuron1.w"]).to(device=device, dtype=torch.float32)
        self.theta2 = self._theta(checkpoint_state["Neuron2.w"]).to(device=device, dtype=torch.float32)
        self.v_threshold = 1.0

    @staticmethod
    def _theta(w: torch.Tensor) -> torch.Tensor:
        w_f = w.detach().float()
        return w_f / torch.sqrt(1.0 + w_f * w_f)

    def begin_trial(self) -> None:
        if self.conv1_sampler is not None:
            self.conv1_sampler.begin_trial()
        if self.conv2_sampler is not None:
            self.conv2_sampler.begin_trial()

    def _conv1_weight(self) -> torch.Tensor:
        if self.mode == "fp32":
            return self.conv1_fp32
        if self.conv1_sampler is not None:
            return self.conv1_sampler.weight_for_timestep()
        assert self.conv1_ideal is not None
        return self.conv1_ideal

    def _conv2_weight(self) -> torch.Tensor:
        if self.mode == "fp32":
            return self.conv2_fp32
        if self.conv2_sampler is not None:
            return self.conv2_sampler.weight_for_timestep()
        assert self.conv2_ideal is not None
        return self.conv2_ideal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, channels, height, width = x.shape
        v1 = torch.zeros((batch, 64, height - 2, width - 2), dtype=x.dtype, device=x.device)
        v2 = torch.zeros((batch, 1, 1, 1), dtype=x.dtype, device=x.device)
        out_last = None
        theta1 = self.theta1.reshape(()) if self.theta1.numel() == 1 else self.theta1
        theta2 = self.theta2.reshape(()) if self.theta2.numel() == 1 else self.theta2

        for t in range(timesteps):
            y1 = F.conv2d(x[:, t], self._conv1_weight(), bias=None, stride=1, padding=0)
            v1 = v1 + y1 - theta1
            spike1 = (v1 >= self.v_threshold).to(x.dtype)
            v1 = v1 - spike1 * self.v_threshold

            y2 = F.conv2d(spike1, self._conv2_weight(), bias=None, stride=1, padding=0)
            v2 = v2 + y2 - theta2
            out_last = v2
            spike2 = (v2 >= self.v_threshold).to(x.dtype)
            v2 = v2 - spike2 * self.v_threshold

        assert out_last is not None
        return out_last


def downsample_curve(fpr: np.ndarray, tpr: np.ndarray, max_points: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    if fpr.size <= max_points:
        return fpr, tpr
    idx = np.unique(np.linspace(0, fpr.size - 1, max_points).astype(np.int64))
    return fpr[idx], tpr[idx]


def evaluate_net(
    name: str,
    net: torch.nn.Module,
    loader: DataLoader,
    labels_buffer: np.ndarray | None,
    device: torch.device,
    seed: int,
    positive_label: str,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    net.begin_trial()
    net.eval()
    scores = np.empty(len(loader.dataset), dtype=np.float32)
    labels = labels_buffer if labels_buffer is not None else np.empty(len(loader.dataset), dtype=np.uint8)
    loss_sum = 0.0
    seen = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (frame, raw_label) in enumerate(loader, 1):
            frame = frame.to(device=device, dtype=torch.float32, non_blocking=False)
            if positive_label == "zero":
                target = 1.0 - raw_label.float()
            elif positive_label == "one":
                target = raw_label.float()
            else:
                raise ValueError(f"unknown positive-label convention: {positive_label}")
            target = target.to(device=device)
            out_vmem = net(frame).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(out_vmem, target.reshape(-1), reduction="mean")
            n = raw_label.numel()
            scores[seen : seen + n] = out_vmem.detach().cpu().numpy()
            if labels_buffer is None:
                labels[seen : seen + n] = target.detach().cpu().numpy().astype(np.uint8)
            loss_sum += float(loss.item()) * n
            seen += n
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == len(loader):
                print(
                    f"{name} batch={batch_idx}/{len(loader)} seen={seen} "
                    f"elapsed_s={time.time() - t0:.1f}",
                    flush=True,
                )

    scores = scores[:seen]
    labels = labels[:seen]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_value = float(sklearn_auc(fpr, tpr))
    best_threshold, best_tpr, best_fpr = best_threshold_from_roc(fpr, tpr, thresholds)
    pred = scores >= best_threshold
    acc = float((pred == labels.astype(bool)).mean())
    result = {
        "name": name,
        "auc": auc_value,
        "acc_at_best_threshold": acc,
        "best_threshold": best_threshold,
        "best_tpr": best_tpr,
        "best_fpr": best_fpr,
        "loss": float(loss_sum / seen),
        "samples": int(seen),
        "positive_labels_after_inversion": int(labels.sum()),
        "negative_labels_after_inversion": int(seen - labels.sum()),
        "elapsed_s": float(time.time() - t0),
    }
    curve = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    return result, labels, scores, curve


def make_net_for_config(
    state: dict[str, torch.Tensor],
    conv1: LayerQuantization,
    conv2: LayerQuantization,
    dist: EmpiricalRRAMDistribution,
    mode: str,
    device: torch.device,
) -> RRAMMappedPSIFNet:
    conv1_sampler = None
    conv2_sampler = None
    if mode == "static_empirical":
        conv1_sampler = RRAMWeightSampler(conv1.q, conv1.scale, dist, device)
        conv2_sampler = RRAMWeightSampler(conv2.q, conv2.scale, dist, device)
    return RRAMMappedPSIFNet(
        checkpoint_state=state,
        conv1_sampler=conv1_sampler,
        conv2_sampler=conv2_sampler,
        conv1_quant=conv1,
        conv2_quant=conv2,
        device=device,
        mode=mode,
    ).to(device)


def weight_error_summary(
    conv1: LayerQuantization,
    conv2: LayerQuantization,
    dist: EmpiricalRRAMDistribution,
    mode: str,
    device: torch.device,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    out: dict[str, dict[str, float]] = {}
    if mode != "static_empirical":
        for item in (conv1, conv2):
            out[item.name] = {
                "rmse_code": 0.0,
                "std_code": 0.0,
                "mean_code": 0.0,
                "max_abs_code": 0.0,
            }
        return out
    for item in (conv1, conv2):
        sampler = RRAMWeightSampler(item.q, item.scale, dist, device)
        sampler.begin_trial()
        q_ideal = torch.as_tensor(item.q, dtype=torch.float32, device=device)
        deltas = []
        for _ in range(samples):
            w = sampler.sample_weight()
            q_eff = w / item.scale
            deltas.append((q_eff - q_ideal).detach().reshape(-1).cpu().numpy())
        delta = np.concatenate(deltas)
        out[item.name] = {
            "rmse_code": float(np.sqrt(np.mean(delta**2))),
            "std_code": float(np.std(delta)),
            "mean_code": float(np.mean(delta)),
            "max_abs_code": float(np.max(np.abs(delta))),
        }
    return out


def config_name(mode: str, trial: int) -> str:
    return mode if trial == 0 else f"{mode}_mc{trial}"


def aggregate_results(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = str(row["config_key"])
        grouped.setdefault(key, []).append(row)
    out = []
    for key, items in grouped.items():
        first = items[0]
        aucs = np.array([float(item["auc"]) for item in items], dtype=np.float64)
        accs = np.array([float(item["acc_at_best_threshold"]) for item in items], dtype=np.float64)
        losses = np.array([float(item["loss"]) for item in items], dtype=np.float64)
        out.append(
            {
                "config_key": key,
                "mode": first["mode"],
                "trials": len(items),
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "loss_mean": float(np.mean(losses)),
                "loss_std": float(np.std(losses)),
                "samples": first["samples"],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_results(aggregate: list[dict[str, float | int | str]], out_dir: Path) -> None:
    labels = [str(row["config_key"]) for row in aggregate]
    auc = np.array([float(row["auc_mean"]) for row in aggregate])
    auc_err = np.array([float(row["auc_std"]) for row in aggregate])
    fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(labels)), 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, auc, yerr=auc_err, color="#4c78a8", alpha=0.86, capsize=3)
    ax.set_ylim(max(0.0, float(auc.min()) - 0.06), min(1.0, float(auc.max()) + 0.025))
    ax.set_ylabel("AUC")
    ax.set_title("TSNF AUC under RRAM mapping")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "nonideal_auc_summary.png", dpi=240)
    plt.close(fig)


def plot_roc_curves(
    curves: dict[str, dict[str, np.ndarray]],
    results: list[dict[str, float | int | str]],
    out_dir: Path,
) -> None:
    auc_lookup = {str(row["name"]): float(row["auc"]) for row in results}
    fig, ax = plt.subplots(figsize=(6.7, 5.8), constrained_layout=True)
    for name, curve in curves.items():
        fpr, tpr = downsample_curve(curve["fpr"], curve["tpr"])
        ax.plot(fpr, tpr, linewidth=1.3, label=f"{name} AUC={auc_lookup.get(name, float('nan')):.4f}")
    ax.plot([0, 1], [0, 1], linestyle=":", color="0.6", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(out_dir / "nonideal_roc_curves.png", dpi=240)
    plt.close(fig)


def save_weight_error_histograms(
    conv1: LayerQuantization,
    conv2: LayerQuantization,
    dist: EmpiricalRRAMDistribution,
    device: torch.device,
    out_dir: Path,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for ax, item in zip(axes, (conv1, conv2)):
        sampler = RRAMWeightSampler(item.q, item.scale, dist, device)
        sampler.begin_trial()
        q_ideal = torch.as_tensor(item.q, dtype=torch.float32, device=device)
        q_eff = (sampler.sample_weight() / item.scale).detach()
        delta = (q_eff - q_ideal).reshape(-1).cpu().numpy()
        ax.hist(delta, bins=80, color="#59a14f", alpha=0.82)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("Effective code error")
        ax.set_ylabel("Count")
        ax.set_title(item.name)
    fig.savefig(out_dir / "static_empirical_weight_code_error_hist.png", dpi=240)
    plt.close(fig)


def file_metadata(path: Path, include_hash: bool) -> dict[str, str | int]:
    resolved = path.resolve()
    metadata: dict[str, str | int] = {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }
    if include_hash:
        metadata["sha256"] = sha256_file(resolved)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TSNF with Int3 and measured static RRAM mappings.")
    default_data = os.environ.get("TSNF_DATASET")
    default_checkpoint = os.environ.get("TSNF_CHECKPOINT")
    parser.add_argument(
        "--data-dir",
        default=default_data,
        required=default_data is None,
        help="NPZ dataset containing spikes.npy and labels.npy (or set TSNF_DATASET).",
    )
    parser.add_argument(
        "--checkpoint",
        default=default_checkpoint,
        required=default_checkpoint is None,
        help="TSNF checkpoint containing the net state dictionary (or set TSNF_CHECKPOINT).",
    )
    parser.add_argument("--rram-csv", default=str(DEFAULT_RRAM_CSV))
    parser.add_argument("--rram-csv-unit", default="kohm", choices=["kohm", "ohm"])
    parser.add_argument("--conv1-q", default=str(DEFAULT_CONV1_Q))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--bits", type=int, default=3, choices=[3], help="Fixed TSNF deployment precision.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional shared npy cache directory for spikes.npy and labels.npy.",
    )
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--mc-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--weight-error-samples", type=int, default=32)
    parser.add_argument(
        "--positive-label",
        choices=["zero", "one"],
        default="zero",
        help="Raw dataset label treated as the positive class. The published TSNF dataset uses zero.",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow legacy pickle checkpoint loading. Use only for a checkpoint from a trusted source.",
    )
    parser.add_argument("--hash-inputs", action="store_true", help="Record SHA-256 hashes of all input files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.data_dir)
    rram_csv_path = Path(args.rram_csv)
    conv1_q_path = Path(args.conv1_q)
    for input_path in (checkpoint_path, dataset_path, rram_csv_path, conv1_q_path):
        if not input_path.is_file():
            raise FileNotFoundError(f"required input not found: {input_path}")

    t0 = time.time()
    dist = load_empirical_rram_distribution(rram_csv_path, unit=args.rram_csv_unit)
    plot_rram_distribution(dist, out_dir)
    checkpoint_state, conv1, conv2 = load_quantized_weights(
        checkpoint_path,
        conv1_q_path,
        bits=args.bits,
        trust_checkpoint=args.trust_checkpoint,
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "npy_cache"
    spikes_path, labels_path = prepare_npy_cache(dataset_path, cache_dir)
    dataset = SpikesLabelsMemmapDataset(spikes_path, labels_path)
    dataset = make_dataset(dataset, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    print(f"dataset_size={len(dataset)} device={device} load_s={time.time() - t0:.2f}", flush=True)

    modes = ["fp32", "ideal_int3", "static_empirical"]
    labels_buffer = None
    results: list[dict[str, float | int | str]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    weight_errors: dict[str, dict[str, dict[str, float]]] = {}
    for mode in modes:
        repeats = 1 if mode in {"fp32", "ideal_int3"} else args.mc_trials
        key = config_name(mode, trial=0)
        weight_errors[key] = weight_error_summary(
            conv1, conv2, dist, mode, device, args.weight_error_samples, seed=args.seed
        )
        if mode == "static_empirical":
            save_weight_error_histograms(conv1, conv2, dist, device, out_dir, seed=args.seed)
        for trial in range(repeats):
            name = config_name(mode, trial=trial if repeats > 1 else 0)
            net = make_net_for_config(checkpoint_state, conv1, conv2, dist, mode, device)
            result, labels, _, curve = evaluate_net(
                name,
                net,
                loader,
                labels_buffer,
                device,
                args.seed + trial,
                args.positive_label,
            )
            if labels_buffer is None:
                labels_buffer = labels
            result.update(
                {
                    "config_key": key,
                    "mode": mode,
                    "trial": trial,
                }
            )
            results.append(result)
            if trial == 0:
                curves[name] = curve
            print(
                f"RESULT {name} auc={result['auc']:.6f} acc={result['acc_at_best_threshold']:.6f} "
                f"loss={result['loss']:.6f}",
                flush=True,
            )

            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = aggregate_results(results)
    plot_accuracy_results(aggregate, out_dir)
    plot_roc_curves(curves, results, out_dir)

    summary = {
        "schema_version": 4,
        "command_line": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "scikit_learn": str(sklearn.__version__),
            "matplotlib": str(matplotlib.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "deterministic_algorithms": True,
            "legacy_pickle_checkpoint_trusted": bool(args.trust_checkpoint),
        },
        "inputs": {
            "checkpoint": file_metadata(checkpoint_path, args.hash_inputs),
            "dataset": file_metadata(dataset_path, args.hash_inputs),
            "rram_csv": file_metadata(rram_csv_path, args.hash_inputs),
            "conv1_codes": file_metadata(conv1_q_path, args.hash_inputs),
        },
        "samples": len(dataset),
        "sample_selection": "first N records in archive order" if args.max_samples > 0 else "all records",
        "positive_raw_label": args.positive_label,
        "bits": args.bits,
        "device": str(device),
        "rram_distribution": dist.summary(),
        "quantization": {
            "conv1": layer_quant_summary(conv1),
            "conv2": layer_quant_summary(conv2),
            "note": (
                "conv1 q codes use the deployed 18x64 matrix; conv2 is per-layer Int3 "
                "MSE quantized from the same checkpoint."
            ),
        },
        "experiment": {
            "mc_trials": args.mc_trials,
            "seed": args.seed,
            "weight_error_samples": args.weight_error_samples,
            "circuit_neuron_path_included": False,
            "release_note": "Circuit-level neuron models will be added in a later release.",
            "accuracy_caveat": (
                "acc_at_best_threshold selects the threshold on this evaluation set and is descriptive only; "
                "AUC is the primary threshold-independent metric."
            ),
        },
        "weight_error_summary": weight_errors,
        "results": results,
        "aggregate": aggregate,
        "outputs": {
            "summary_json": str(out_dir / "summary.json"),
            "results_csv": str(out_dir / "results.csv"),
            "aggregate_csv": str(out_dir / "aggregate.csv"),
            "rram_distribution_png": str(out_dir / "measured_rram_state_distribution.png"),
            "auc_summary_png": str(out_dir / "nonideal_auc_summary.png"),
            "roc_curves_png": str(out_dir / "nonideal_roc_curves.png"),
        },
        "elapsed_s": float(time.time() - t0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "results.csv", results)
    write_csv(out_dir / "aggregate.csv", aggregate)
    print("SUMMARY_JSON_BEGIN")
    print(json.dumps(summary, indent=2))
    print("SUMMARY_JSON_END")


if __name__ == "__main__":
    main()
