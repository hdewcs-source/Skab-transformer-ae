# Transformer Autoencoder for Anomaly Detection on SKAB

Unsupervised anomaly detection on multivariate industrial sensor time series with a Transformer autoencoder, evaluated on the valve1 part of the [Skoltech Anomaly Benchmark (SKAB)](https://github.com/waico/SKAB). The repository also documents the limits of the benchmark and of the first version of this experiment.

## Overview

- Task: flag anomalous windows in an industrial water-circulation testbed with 8 sensor channels (two accelerometers, current, pressure, temperature, thermocouple, voltage, volume flow rate).
- Method: a model is trained on normal data only. A window is scored by its reconstruction error, and a high error means an anomaly.
- Baselines: Isolation Forest, PCA reconstruction, a predictor that flags everything, and a diagnostic rule that uses only the position inside a recording.

## Data

```bash
git clone --depth 1 https://github.com/waico/SKAB.git data/SKAB
```

- Training: `data/anomaly-free/anomaly-free.csv`, split chronologically into 80 percent for training and 20 percent for validation.
- Test: the 16 recordings in `data/valve1/`. Each recording is scaled and windowed separately, so no window crosses a recording boundary.
- Preprocessing: Min-Max scaling fitted on the training part, sliding window of 30 steps, window label equal to the label of its last step.
- The SKAB copy on Kaggle used in the initial notebook has slightly different row counts from the GitHub version. Numbers in the notebook and in this README are therefore not directly comparable.

## Models

`src/transformer_ae.py` provides two decoder variants, because the initial notebook and the first written report describe different decoders.

| Variant | Decoder target | Note |
| --- | --- | --- |
| `original` | projected input window | what the initial notebook implements, the decoder can read the input directly |
| `latent` | encoder output | the formulation TransformerDecoder(z, z) |

Positional encoding is off by default to match the initial notebook. Without it, self-attention treats a window as an unordered set of time steps. It can be enabled with `--pos-enc`.

## Evaluation protocol

No test label is used to choose a threshold.

- Threshold-free: AUROC and AUPRC on the test scores.
- Threshold-based: precision, recall and F1 at a threshold that comes from normal data only. The default (`--calibration recording`) normalises every test recording with its own first 400 rows, which SKAB defines as anomaly free, and uses the 99th percentile of the pooled calibration scores. The alternative (`--calibration normal-file`) uses held-out windows of the anomaly-free recording.
- Oracle best F1: the best F1 over all thresholds, chosen with the test labels. It is an upper bound and is shown for reference only.
- Seeds: mean and standard deviation over several seeds.

## Results

Classical baselines and diagnostics, computed with `python src/run_experiment.py --data-root data/SKAB/data --skip-ae --seeds 0 1 2`:

| Method | AUROC | AUPRC | F1 | Precision | Recall | Oracle best F1 |
| --- | --- | --- | --- | --- | --- | --- |
| All anomalous (trivial) | - | - | 0.526 | 0.357 | 1.000 | - |
| Fixed-position rule (diagnostic) | - | - | 0.981 | 0.974 | 0.988 | - |
| Isolation Forest | 0.488 (0.003) | 0.350 (0.002) | 0.021 (0.005) | 0.347 (0.043) | 0.011 (0.003) | 0.526 (0.000) |
| PCA reconstruction | 0.596 (0.000) | 0.493 (0.000) | 0.386 (0.000) | 0.640 (0.000) | 0.276 (0.000) | 0.526 (0.000) |

Transformer AE results are produced by the same script without `--skip-ae` and are written to `results/results.md`.

### Observations about the benchmark

- In all 16 valve1 recordings the anomaly is one contiguous block of 337 to 405 rows that starts between row 566 and row 578. A rule that flags rows 570 to 969 and uses no sensor signal reaches F1 0.981. A high F1 on valve1 alone therefore says little about a model, and AUROC, AUPRC and comparison with such diagnostics are more informative.
- An Isolation Forest or PCA fitted on the single anomaly-free recording cannot separate anomalies in the test recordings. With a 99th percentile threshold taken from the anomaly-free recording (`--calibration normal-file`), both baselines flag every test window.

## Initial exploration

`notebooks/01_initial_exploration.ipynb` is the first version of the experiment. It reports F1 0.861 for a Transformer autoencoder against 0.516 for Isolation Forest. Two limitations apply to those numbers.

1. The threshold was the 70th percentile of the test reconstruction errors, and the percentile was chosen by looking at the test F1. The fraction of flagged windows therefore follows the test anomaly ratio, which is not available in deployment.
2. The Isolation Forest baseline flagged all windows. An F1 of 0.516 is exactly the F1 of always answering anomalous when 35 percent of the windows are anomalous.

## Reproduce

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/waico/SKAB.git data/SKAB
python src/run_experiment.py --data-root data/SKAB/data --seeds 0 1 2 3 4
```

Useful options: `--variants original latent`, `--pos-enc`, `--calibration normal-file`, `--label-mode any`, `--quantile 95`, `--skip-ae`.

## Repository layout

```
src/skab_data.py          data loading, windowing, per-recording handling
src/transformer_ae.py     model, training, reconstruction scores
src/baselines.py          Isolation Forest and PCA scores
src/metrics.py            thresholds, AUROC, AUPRC, oracle F1, diagnostics
src/run_experiment.py     command line entry point
notebooks/                initial exploration, kept for reference
results/                  generated tables and scores
```

## Limitations

- SKAB valve1 has a fixed anomaly position in every recording, see above.
- A single dataset and a single training recording are used.
- Windows are scored independently, and no event-level metric is reported.
