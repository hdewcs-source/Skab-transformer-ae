"""Run the SKAB experiments with a threshold chosen without test labels.

Example
-------
git clone --depth 1 https://github.com/waico/SKAB.git data/SKAB
python src/run_experiment.py --data-root data/SKAB/data --seeds 0 1 2 3 4

Use --skip-ae to run only the classical baselines (no PyTorch needed).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from baselines import isolation_forest_scores, pca_reconstruction_scores
from metrics import (
    all_anomalous_metrics,
    evaluate_recording_calibrated,
    evaluate_scores,
    position_prior_metrics,
)
from skab_data import load_skab

METRIC_KEYS = ["auroc", "auprc", "f1", "precision", "recall", "flagged_fraction", "oracle_best_f1"]


def summarize(runs):
    """Mean and standard deviation over seeds for every metric."""
    return {
        key: {
            "mean": float(np.mean([r[key] for r in runs])),
            "std": float(np.std([r[key] for r in runs])),
        }
        for key in METRIC_KEYS
    }


def to_markdown(summary, anomaly_ratio, quantile, n_seeds, calibration, n_calib):
    if calibration == "recording":
        how = (f"Scores are z-normalised per recording with its first {n_calib} rows, which SKAB "
               f"defines as anomaly free. Threshold: {quantile}th percentile of the pooled "
               f"calibration scores.")
    else:
        how = (f"Threshold: {quantile}th percentile of scores on held-out normal windows "
               f"of the anomaly-free recording.")
    lines = [
        f"Test anomaly ratio: {anomaly_ratio:.3f}. {how} Mean and standard deviation over "
        f"{n_seeds} seeds.",
        "",
        "| Method | AUROC | AUPRC | F1 | Precision | Recall | Oracle best F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, s in summary.items():
        def cell(k):
            if "mean" not in s.get(k, {}):
                return "-"
            return f"{s[k]['mean']:.3f} ({s[k]['std']:.3f})"
        lines.append(
            f"| {name} | {cell('auroc')} | {cell('auprc')} | {cell('f1')} | "
            f"{cell('precision')} | {cell('recall')} | {cell('oracle_best_f1')} |"
        )
    lines += [
        "",
        "Oracle best F1 uses the test labels to pick the best threshold and is an upper bound, "
        "not a realistic result. The fixed-position rule flags rows 570 to 969 of every recording "
        "and uses no sensor signal. It shows how much of SKAB valve1 can be solved by position alone.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="folder containing anomaly-free/ and valve1/")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--label-mode", choices=["last", "any"], default="last")
    parser.add_argument("--quantile", type=float, default=99.0)
    parser.add_argument("--calibration", choices=["recording", "normal-file"], default="recording",
                        help="recording: per-recording normalisation with its first --n-calib rows; "
                             "normal-file: threshold from held-out anomaly-free data")
    parser.add_argument("--n-calib", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--variants", nargs="+", default=["original", "latent"],
                        choices=["original", "latent"])
    parser.add_argument("--pos-enc", action="store_true", help="add sinusoidal positional encoding")
    parser.add_argument("--score", choices=["max_dim", "mean"], default="max_dim")
    parser.add_argument("--skip-ae", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_skab(args.data_root, window=args.window, label_mode=args.label_mode)
    x_train, x_val, x_test, y = data["x_train"], data["x_val"], data["x_test"], data["y_test"]
    print(f"train {x_train.shape}, val {x_val.shape}, test {x_test.shape}, "
          f"anomaly ratio {y.mean():.3f}, recordings {data['n_recordings']}")

    def as_summary(d):
        return {k: {"mean": v, "std": 0.0} for k, v in d.items()}

    def evaluate(val_scores, test_scores):
        if args.calibration == "recording":
            return evaluate_recording_calibrated(
                test_scores, y, data["recording_id"], data["end_index"], args.n_calib, args.quantile
            )
        return evaluate_scores(val_scores, test_scores, y, args.quantile)

    summary = {
        "All anomalous (trivial)": as_summary(all_anomalous_metrics(y)),
        "Fixed-position rule (diagnostic)": as_summary(position_prior_metrics(y, data["end_index"])),
    }
    raw = {}

    runs = []
    for seed in args.seeds:
        v, t = isolation_forest_scores(x_train, x_val, x_test, seed=seed)
        runs.append(evaluate(v, t))
    summary["Isolation Forest"] = summarize(runs)
    raw["Isolation Forest"] = runs

    v, t = pca_reconstruction_scores(x_train, x_val, x_test)
    pca_run = evaluate(v, t)
    summary["PCA reconstruction"] = summarize([pca_run])
    raw["PCA reconstruction"] = [pca_run]

    if not args.skip_ae:
        from transformer_ae import reconstruction_scores, train_model

        for variant in args.variants:
            name = f"Transformer AE ({variant}{', pos-enc' if args.pos_enc else ''})"
            runs = []
            for seed in args.seeds:
                print(f"{name}, seed {seed}")
                model = train_model(x_train, variant=variant, pos_enc=args.pos_enc,
                                    epochs=args.epochs, seed=seed)
                v = reconstruction_scores(model, x_val, score=args.score)
                t = reconstruction_scores(model, x_test, score=args.score)
                runs.append(evaluate(v, t))
                if seed == args.seeds[0]:
                    np.save(out_dir / f"scores_{variant}_seed{seed}.npy", t)
            summary[name] = summarize(runs)
            raw[name] = runs

    config = vars(args)
    config["anomaly_ratio"] = float(y.mean())
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": config, "summary": summary, "runs": raw}, f, indent=2)
    table = to_markdown(summary, float(y.mean()), args.quantile, len(args.seeds),
                        args.calibration, args.n_calib)
    (out_dir / "results.md").write_text(table + "\n")
    print()
    print(table)


if __name__ == "__main__":
    main()
