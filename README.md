# RED-SNN simulator

This repository contains the first public version of RED-SNN, an SNN simulator
for event camera denoising. This release focuses on TSNF inference and its
RRAM-aware weight-mapping path.

The trained checkpoint, evaluation dataset, circuit-level neuron models, and
the remaining simulator components are not included yet. They will come soon.

## What is included

- TSNF inference from a trained checkpoint
- signed per-layer Int3 weight quantization
- differential 2T2R mapping with measured LRS and HRS conductance data
- static-device Monte Carlo evaluation
- JSON, CSV, and figure outputs for checking each run

The script compares three paths: the original FP32 model, ideal Int3 weights,
and static empirical RRAM weights. Circuit-level neuron simulation, circuit
calibration, and PPA analysis are outside the scope of this release. The
evaluation assumptions are listed in [the methodology](docs/methodology.md).

## Setup

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run

Start with the smoke profile to check the environment:

```bash
python quickstart.py \
  --checkpoint /path/to/training_checkpoint_AUC_max.pth \
  --dataset /path/to/spikes_labels_dataset.npz
```

CUDA is selected automatically when available. Results are written to
`results/` by default.

| Profile | Records | RRAM trials | Intended use |
|---|---:|---:|---|
| `smoke` | 4,096 | 1 | Quick installation check |
| `standard` | 100,000 | 2 | Routine comparison |
| `full` | all | 3 | Full evaluation |

For example:

```bash
python quickstart.py --checkpoint checkpoint.pth --dataset dataset.npz --profile standard
python quickstart.py --checkpoint checkpoint.pth --dataset dataset.npz --profile full --device cpu
```

## Input files

The checkpoint must contain a `net` state dictionary with `conv1.weight`,
`conv2.weight`, `Neuron1.w`, and `Neuron2.w`.

The dataset must be an NPZ archive containing:

- `spikes.npy` with shape `[N, 32, 2, 7, 7]`
- binary `labels.npy` with shape `[N]`

Neither file is downloaded automatically. Safe weights-only checkpoint loading
is used by default. Use `--trust-checkpoint` only for an older checkpoint whose
source you have checked.

For reproducible records, `--hash-inputs` adds SHA-256 hashes to `summary.json`.
The `--cache-dir PATH` option can be used to reuse extracted dataset arrays.

## Results

Each run produces a `summary.json` file, per-trial and aggregate CSV files, ROC
and AUC plots, measured-state distributions, and effective-code error plots.
The summary also records the environment, seed, input metadata, quantization
settings, RRAM statistics, and final metrics.

AUC is the main reported metric. Accuracy at the best ROC threshold is included
as a descriptive value because the threshold is selected on the evaluated
records.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

These tests cover RRAM data parsing and static persistence, Int3 quantization,
and dataset-cache provenance.

## Data and license

The included conductance array is documented in
[the measurement data card](docs/rram_measurement_data_card.md). Its provenance
and redistribution notes should be checked before republishing the data.

The source code is released under the [MIT License](LICENSE). Checkpoints,
evaluation datasets, generated results, and unrelated hardware projects are
not part of this repository.
