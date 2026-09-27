from __future__ import annotations

import hashlib
import json
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpikesLabelsMemmapDataset(Dataset):
    def __init__(self, spikes_path: str | Path, labels_path: str | Path) -> None:
        self.spikes_path = Path(spikes_path)
        self.labels_path = Path(labels_path)
        self.spikes = np.load(self.spikes_path, mmap_mode="r")
        self.labels = np.load(self.labels_path, mmap_mode="r")
        if self.spikes.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"spikes/labels length mismatch: {self.spikes.shape[0]} vs {self.labels.shape[0]}"
            )
        if self.spikes.ndim != 5 or tuple(self.spikes.shape[1:]) != (32, 2, 7, 7):
            raise ValueError(
                "expected spikes with shape [N, 32, 2, 7, 7], "
                f"got {tuple(self.spikes.shape)}"
            )
        if self.labels.ndim != 1:
            raise ValueError(f"expected one-dimensional labels, got shape {tuple(self.labels.shape)}")

        unique_labels = np.unique(self.labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"expected binary labels 0/1, got {unique_labels.tolist()}")

    def __len__(self) -> int:
        return int(self.spikes.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame = torch.from_numpy(np.array(self.spikes[index], dtype=np.float32, copy=True))
        label = torch.as_tensor(int(self.labels[index]), dtype=torch.uint8)
        return frame, label


def prepare_npy_cache(npz_path: str | Path, cache_dir: str | Path) -> tuple[Path, Path]:
    npz_path = Path(npz_path)
    cache_dir = Path(cache_dir)
    if not npz_path.is_file():
        raise FileNotFoundError(f"dataset archive not found: {npz_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"spikes.npy": cache_dir / "spikes.npy", "labels.npy": cache_dir / "labels.npy"}
    manifest_path = cache_dir / "cache_manifest.json"

    with zipfile.ZipFile(npz_path, "r") as archive:
        names = set(archive.namelist())
        missing = [name for name in wanted if name not in names]
        if missing:
            raise KeyError(f"npz missing arrays: {missing}; found {sorted(names)}")
        source_stat = npz_path.stat()
        expected_manifest = {
            "source_path": str(npz_path.resolve()),
            "source_size_bytes": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "members": {
                name: {
                    "crc32": int(archive.getinfo(name).CRC),
                    "size_bytes": int(archive.getinfo(name).file_size),
                }
                for name in wanted
            },
        }

        if all(path.is_file() for path in wanted.values()):
            if manifest_path.is_file():
                cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if cached_manifest == expected_manifest:
                    return wanted["spikes.npy"], wanted["labels.npy"]
            elif all(
                _crc32_file(path) == expected_manifest["members"][name]["crc32"]
                for name, path in wanted.items()
            ):
                manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
                return wanted["spikes.npy"], wanted["labels.npy"]

        print(f"extracting npz arrays to memmap cache: {cache_dir}", flush=True)
        for member, out_path in wanted.items():
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            with archive.open(member) as source, tmp_path.open("wb") as target:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            tmp_path.replace(out_path)
            print(f"cached {member} -> {out_path}", flush=True)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
    return wanted["spikes.npy"], wanted["labels.npy"]


def _crc32_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def quantize_vec_signed_symmetric(
    values: np.ndarray,
    bits: int,
    method: str = "mse",
    candidates: int = 160,
) -> tuple[np.ndarray, dict[str, float]]:
    values = values.astype(np.float32, copy=False)
    if bits < 2:
        raise ValueError(f"signed symmetric quantization requires bits >= 2, got {bits}")

    absmax = float(np.max(np.abs(values))) if values.size else 0.0
    if absmax == 0.0:
        return values.copy(), {"scale": 1.0, "clip": 0.0, "mse": 0.0, "sat_frac": 0.0}

    qmax = (1 << (bits - 1)) - 1
    if method == "max":
        clip_values = np.array([absmax], dtype=np.float32)
    elif method == "mse":
        lower = max(absmax / qmax, float(np.percentile(np.abs(values), 60)))
        lower = min(lower, absmax)
        linear = np.linspace(lower, absmax, candidates, dtype=np.float32)
        percentiles = np.percentile(np.abs(values), [70, 80, 90, 95, 97.5, 99, 99.5, 99.9])
        clip_values = np.unique(np.concatenate([linear, percentiles.astype(np.float32), [absmax]]))
        clip_values = clip_values[(clip_values > 0) & (clip_values <= absmax)]
    else:
        raise ValueError(f"unknown quantization method: {method}")

    best_dequantized: np.ndarray | None = None
    best_stats: dict[str, float] | None = None
    for clip_value in clip_values:
        clip = float(clip_value)
        scale = clip / qmax
        q = np.round(np.clip(values, -clip, clip) / scale)
        q = np.clip(q, -qmax, qmax)
        dequantized = (q * scale).astype(np.float32)
        mse = float(np.mean((dequantized - values) ** 2))
        if best_stats is None or mse < best_stats["mse"]:
            best_dequantized = dequantized
            best_stats = {
                "scale": scale,
                "clip": clip,
                "mse": mse,
                "sat_frac": float(np.mean(np.abs(values) >= clip)),
            }

    if best_dequantized is None or best_stats is None:
        raise RuntimeError("failed to determine a quantization scale")
    return best_dequantized, best_stats


def best_threshold_from_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float, float]:
    distance = np.sqrt((tpr - 1.0) ** 2 + fpr**2)
    index = int(np.argmin(distance))
    return float(thresholds[index]), float(tpr[index]), float(fpr[index])


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
