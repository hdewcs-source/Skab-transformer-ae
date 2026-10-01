Test anomaly ratio: 0.357. Scores are z-normalised per recording with its first 400 rows, which SKAB defines as anomaly free. Threshold: 99.0th percentile of the pooled calibration scores. Mean and standard deviation over 3 seeds.

| Method | AUROC | AUPRC | F1 | Precision | Recall | Oracle best F1 |
| --- | --- | --- | --- | --- | --- | --- |
| All anomalous (trivial) | - | - | 0.526 (0.000) | 0.357 (0.000) | 1.000 (0.000) | - |
| Fixed-position rule (diagnostic) | - | - | 0.981 (0.000) | 0.974 (0.000) | 0.988 (0.000) | - |
| Isolation Forest | 0.488 (0.003) | 0.350 (0.002) | 0.021 (0.005) | 0.347 (0.043) | 0.011 (0.003) | 0.526 (0.000) |
| PCA reconstruction | 0.596 (0.000) | 0.493 (0.000) | 0.386 (0.000) | 0.640 (0.000) | 0.276 (0.000) | 0.526 (0.000) |

Oracle best F1 uses the test labels to pick the best threshold and is an upper bound, not a realistic result. The fixed-position rule flags rows 570 to 969 of every recording and uses no sensor signal. It shows how much of SKAB valve1 can be solved by position alone.
