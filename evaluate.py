"""Re-evaluate the saved deployed model on the saved held-out test split.

    python evaluate.py

Loads models/predictx_best_model.joblib and data/processed/test.csv (written by
train.py), recomputes every metric, regenerates the deployed-model figures and checks
that results match the ones recorded at training time (reproducibility check).
"""
from __future__ import annotations

import sys

import pandas as pd

from src.data_loader import standardize_columns
from src.evaluation import (classification_report_dict, compute_metrics, measure_inference_time, plot_calibration,
                            plot_class_distribution, plot_confusion_matrices, plot_pr_curves, plot_roc_curves)
from src.inference import PredictXPredictor
from src.preprocessing import get_X_y
from src.utils import get_logger, load_config, load_json, resolve_path, save_json


def main() -> int:
    cfg = load_config()
    log = get_logger("predictx.evaluate", resolve_path(cfg["paths"]["logs_dir"]) / "evaluate.log")
    test_path = resolve_path(cfg["paths"]["processed_dir"]) / "test.csv"
    if not test_path.exists():
        log.error("No saved test split at %s — run train.py first", test_path)
        return 2
    predictor = PredictXPredictor(cfg=cfg)
    test = standardize_columns(pd.read_csv(test_path))
    X, y = get_X_y(test)
    p = predictor.predict_proba(X)
    thr = predictor.threshold
    m = compute_metrics(y, p, thr)
    name = predictor.metadata["deployed_label"]
    log.info("Deployed model: %s | threshold %.4f", name, thr)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate",
              "false_negative_rate", "brier", "ece"]:
        log.info("  %-20s %.4f", k, m[k])
    log.info("  confusion matrix  TN=%d FP=%d FN=%d TP=%d", m["tn"], m["fp"], m["fn"], m["tp"])
    timing = measure_inference_time(predictor.model, X)
    log.info("  inference: %.4f ms/sample (batch), %.2f ms single-row", timing["ms_per_sample_batch"],
             timing["single_row_latency_ms"])

    figs = resolve_path(cfg["paths"]["figures_dir"])
    curves = {name: (y.to_numpy(), p, thr)}
    plot_confusion_matrices(curves, figs / "evaluate_confusion_matrix.png", title="evaluate.py — test set")
    plot_roc_curves(curves, figs / "evaluate_roc.png")
    plot_pr_curves(curves, figs / "evaluate_pr.png")
    plot_calibration(curves, figs / "evaluate_calibration.png")
    plot_class_distribution(y, p, thr, figs / "evaluate_class_distribution.png")
    save_json({"model": name, "metrics": m, "inference": timing,
               "classification_report": classification_report_dict(y, p, thr)},
              resolve_path(cfg["paths"]["reports_dir"]) / "evaluate_report.json")

    recorded = load_json(resolve_path(cfg["paths"]["reports_dir"]) / "final_test_metrics.json")["metrics"]
    diffs = {k: abs(recorded[k] - m[k]) for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]}
    if max(diffs.values()) > 1e-9:
        log.warning("Metrics differ from training-time record: %s", diffs)
        return 1
    log.info("Reproducibility check passed: metrics identical to the training-time record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
