# TSNF-RRAM Evaluation Methodology

This document defines the algorithm-level accuracy path implemented by
`evaluate_tsnf_rram.py`. Circuit-level neuron models, PPA analysis, EMD, RTL,
and FPGA/ASIC implementation are outside the scope of this initial release.

## Inputs And Network

The checkpoint `net` dictionary must contain:

| Tensor | Expected shape |
|---|---|
| `conv1.weight` | `[64, 2, 3, 3]` |
| `conv2.weight` | `[1, 64, 5, 5]` |
| `Neuron1.w` | scalar |
| `Neuron2.w` | scalar |

The evaluation archive contains `spikes.npy` with shape `[N, 32, 2, 7, 7]`
and one-dimensional binary `labels.npy`. Raw label 0 is the default positive
class.

## Quantization And RRAM Mapping

Both convolutional layers use MSE-optimized, symmetric per-layer signed Int3
quantization. Conv1 loads the deployed `18 x 64` code matrix from
`data/ptq_conv1_int3_per_layer_code_matrix.csv`; evaluation stops if it differs
from direct quantization of the supplied checkpoint. Conv2 is quantized from
the same checkpoint.

Each magnitude bit, weighted 1 or 2, maps to a differential 2T2R pair. An
active positive or negative branch uses LRS while its opposite branch uses
HRS; an inactive pair uses HRS/HRS. Effective code is normalized by the
difference between median measured LRS and HRS conductance.

`data/read_res_arr_28nm.csv` is interpreted as equivalent 1T1R read resistance
in kOhm. Invalid values are removed, and deterministic two-cluster k-means in
log-resistance space separates the unlabeled LRS and HRS populations.

## Evaluation Cases

`fp32` uses the original checkpoint weights. `ideal_int3` uses dequantized
Int3 weights. `static_empirical` samples one conductance for every mapped cell
at the start of each trial and holds that realization fixed for all records
and timesteps. The latter represents population-level state variation across
a programmed array, not repeated-read noise.

## Profiles And Outputs

`quickstart.py` supplies three fixed profiles:

| Profile | Records | Static-RRAM trials |
|---|---:|---:|
| `smoke` | 4,096 | 1 |
| `standard` | 100,000 | 2 |
| `full` | all | 3 |

Partial profiles use the first records in archive order; they are not random
or stratified samples. Each run records per-trial metrics, aggregates,
environment details, input metadata, quantization, and RRAM statistics. AUC is
primary; accuracy at the best ROC threshold is descriptive only.

A 100,000-record regression on the local 10 Hz benchmark produced AUC values
of 0.982255 for FP32, 0.985760 for ideal Int3, and 0.979871 for two-trial
static empirical RRAM. Reproducing these values requires the exact checkpoint
and dataset, which will be provided in a later release.

## Model Boundary

The initial release includes signed Int3 quantization, differential bit
slicing, and state-level empirical conductance variation. It does not include
array line resistance, IR drop, access-device nonlinearity, circuit-level
neuron transfer, current-mirror mismatch, comparator error, capacitor
mismatch, circuit noise, supply noise, PVT variation, retention, endurance,
or calibrated fault rates.

Circuit-level neuron models and the remaining simulator components will come
soon. Results from this version must not be presented as covering those
effects.
