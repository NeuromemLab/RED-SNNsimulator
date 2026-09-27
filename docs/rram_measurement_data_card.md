# RRAM Measurement Data Card

## Included Data

- File: `data/read_res_arr_28nm.csv`
- Source described in project records: authors' 28 nm 1T1R test array
- Nominal array size: 128 x 256 cells
- CSV shape: 16 x 32, containing 512 entries
- Valid positive entries: 480
- Simulator unit: equivalent full 1T1R read-path resistance in kOhm
- Population interpretation: measurements from different 1T1R devices, not
  repeated-read traces from one cell
- Inferred clusters: 276 LRS and 204 HRS measurements

The source file does not store state labels. The evaluator separates LRS and
HRS with deterministic two-cluster k-means in log-resistance space.

## Information Still Required For Public Redistribution

- device stack, material system, and fabrication source;
- number of distinct measured cells and the sampling procedure;
- forming, SET, RESET, and compliance conditions;
- read voltage, pulse width, access-transistor bias, and temperature;
- instrumentation, extraction method, and pre-export filtering;
- measurement identifier, responsible laboratory, and permission to publish;
- an explicit license for redistribution of this CSV.

Until these items are confirmed, the supported claim is limited to accuracy
under the supplied empirical state-level resistance distribution. The data do
not establish cycle-to-cycle read noise, retention, endurance, or a complete
process-variation model.
