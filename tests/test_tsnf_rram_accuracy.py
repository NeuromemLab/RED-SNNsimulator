from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evaluate_tsnf_rram import (
    RRAMWeightSampler,
    load_empirical_rram_distribution,
    quantize_layer_per_tensor,
)
from tsnf_rram_eval_utils import SpikesLabelsMemmapDataset, prepare_npy_cache


class TSNFRRAMEvaluationTests(unittest.TestCase):
    def test_empirical_distribution_discards_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "states.csv"
            np.savetxt(csv_path, np.array([[0.0, 10.0, 12.0], [100.0, 120.0, np.nan]]), delimiter=",")
            distribution = load_empirical_rram_distribution(csv_path, unit="kohm")

        self.assertEqual(distribution.raw_count, 6)
        self.assertEqual(distribution.discarded_count, 2)
        self.assertEqual(distribution.resistance_ohm_lrs.size, 2)
        self.assertEqual(distribution.resistance_ohm_hrs.size, 2)
        self.assertGreater(distribution.delta_g_ref_s, 0.0)

    def test_static_empirical_weight_is_fixed_within_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "states.csv"
            np.savetxt(csv_path, np.array([[10.0, 11.0, 100.0, 110.0]]), delimiter=",")
            distribution = load_empirical_rram_distribution(csv_path, unit="kohm")

        sampler = RRAMWeightSampler(
            np.array([[-3, -1, 0, 1, 3]], dtype=np.int16),
            scale=0.25,
            dist=distribution,
            device=torch.device("cpu"),
        )
        torch.manual_seed(7)
        sampler.begin_trial()
        first = sampler.weight_for_timestep()
        second = sampler.weight_for_timestep()
        self.assertTrue(torch.equal(first, second))

    def test_int3_quantization_range(self) -> None:
        weights = np.linspace(-2.0, 2.0, 101, dtype=np.float32).reshape(1, 101)
        quantized = quantize_layer_per_tensor(weights, bits=3)
        self.assertGreaterEqual(int(quantized.q.min()), -3)
        self.assertLessEqual(int(quantized.q.max()), 3)
        np.testing.assert_allclose(quantized.deq, quantized.q.astype(np.float32) * quantized.scale)

    def test_npz_cache_is_bound_to_source_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "dataset.npz"
            cache = root / "cache"
            spikes = np.zeros((2, 32, 2, 7, 7), dtype=np.uint8)
            labels = np.array([0, 1], dtype=np.uint8)
            np.savez(archive, spikes=spikes, labels=labels)

            spikes_path, labels_path = prepare_npy_cache(archive, cache)
            dataset = SpikesLabelsMemmapDataset(spikes_path, labels_path)
            self.assertEqual(len(dataset), 2)
            self.assertTrue((cache / "cache_manifest.json").is_file())

            spikes[0, 0, 0, 0, 0] = 1
            np.savez(archive, spikes=spikes, labels=labels)
            del dataset
            gc.collect()
            spikes_path, labels_path = prepare_npy_cache(archive, cache)
            refreshed = SpikesLabelsMemmapDataset(spikes_path, labels_path)
            self.assertEqual(float(refreshed[0][0][0, 0, 0, 0]), 1.0)
            del refreshed
            gc.collect()


if __name__ == "__main__":
    unittest.main()
