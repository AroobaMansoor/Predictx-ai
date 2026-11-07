"""Metrics, threshold selection, calibration statistics and evaluation figures."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

PALETTE = ["#1F6FEB", "#00B4D8", "#0B1F3A", "#F59E0B", "#10B981", "#E5484D", "#8B5CF6"]


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "ece": expected_calibration_error(y_true, proba),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def classification_report_dict(y_true, proba, threshold: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return classification_report(y_true, pred, target_names=["No Failure", "Failure"], output_dict=True,
                                 zero_division=0)


def find_best_threshold(y_true, proba, strategy: str = "f1", fixed_value: float = 0.5) -> float:
    """Decision threshold chosen on VALIDATION data (never on test)."""
    if strategy == "fixed":
        return float(fixed_value)
    if strategy != "f1":
        raise ValueError(f"Unknown threshold strategy '{strategy}'")
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best = int(np.nanargmax(f1))
    return float(thr[best])


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true); proba = np.asarray(proba)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(proba[m].mean() - y_true[m].mean())
    return float(ece)


def measure_inference_time(model, X: pd.DataFrame, repeats: int = 3, single_rows: int = 30) -> dict:
    """Batch throughput and single-row latency of predict_proba."""
    batch = []
    for _ in range(repeats):
        t0 = time.perf_counter(); model.predict_proba(X); batch.append(time.perf_counter() - t0)
    single = []
    for i in range(min(single_rows, len(X))):
        row = X.iloc[[i]]
        t0 = time.perf_counter(); model.predict_proba(row); single.append(time.perf_counter() - t0)
    b = float(np.median(batch))
    return {"batch_size": len(X), "batch_time_s": b, "ms_per_sample_batch": 1000 * b / len(X),
            "single_row_latency_ms": 1000 * float(np.median(single))}


# ----------------------------------------------------------------- figures
def _style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.titleweight": "bold", "savefig.dpi": 150, "savefig.bbox": "tight"})


def plot_confusion_matrices(results: dict, out_path: str | Path, title: str = "Confusion matrices (test set)"):
    """results: {model_name: (y_true, proba, threshold)}"""
    _style()
    n = len(results); cols = min(3, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, (y, p, t)) in zip(axes, results.items()):
        cm = confusion_matrix(y, (np.asarray(p) >= t).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
        ax.set_title(f"{name}\n(threshold = {t:.3f})", fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold"); fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


def plot_roc_curves(results: dict, out_path, title="ROC curves (test set)"):
    _style(); fig, ax = plt.subplots(figsize=(7, 6))
    for (name, (y, p, _)), c in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)"); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9); fig.savefig(out_path); plt.close(fig)


def plot_pr_curves(results: dict, out_path, title="Precision-recall curves (test set)"):
    _style(); fig, ax = plt.subplots(figsize=(7, 6))
    base = None
    for (name, (y, p, _)), c in zip(results.items(), PALETTE):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=c, lw=2, label=f"{name} (AP = {average_precision_score(y, p):.3f})")
        base = np.mean(y)
    if base is not None:
        ax.axhline(base, color="k", ls="--", lw=1, label=f"No-skill ({base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.set_ylim(0, 1.03); ax.legend(loc="lower left", fontsize=9); fig.savefig(out_path); plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, out_path, metrics=("precision", "recall", "f1", "pr_auc", "roc_auc"),
                          title="Model comparison (test set, validation-tuned thresholds)"):
    _style()
    d = df.set_index("model")[list(metrics)]
    fig, ax = plt.subplots(figsize=(12, 5))
    d.plot(kind="bar", ax=ax, color=PALETTE[:len(metrics)], width=0.8, edgecolor="white")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score"); ax.set_xlabel(""); ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7, padding=1)
    ax.legend(ncol=len(metrics), loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(out_path); plt.close(fig)


def plot_calibration(results: dict, out_path, n_bins: int = 10, title="Probability calibration (reliability diagram)"):
    _style(); fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    for (name, (y, p, _)), c in zip(results.items(), PALETTE):
        frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax[0].plot(mean, frac, "o-", color=c, lw=1.8, ms=4,
                   label=f"{name} (Brier {brier_score_loss(y, p):.4f}, ECE {expected_calibration_error(y, p):.4f})")
        ax[1].hist(p, bins=40, range=(0, 1), histtype="step", color=c, lw=1.5, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax[0].set_xlabel("Mean predicted probability"); ax[0].set_ylabel("Observed failure frequency")
    ax[0].set_title(title); ax[0].legend(fontsize=8)
    ax[1].set_yscale("log"); ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Count (log)")
    ax[1].set_title("Predicted probability histogram")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_class_distribution(y_true, proba, threshold, out_path, title="Actual vs predicted class distribution (test)"):
    _style()
    pred = (np.asarray(proba) >= threshold).astype(int)
    d = pd.DataFrame({"Actual": pd.Series(np.asarray(y_true)).value_counts().reindex([0, 1], fill_value=0).values,
                      "Predicted": pd.Series(pred).value_counts().reindex([0, 1], fill_value=0).values},
                     index=["No Failure", "Failure"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d.plot(kind="bar", ax=ax, color=["#0B1F3A", "#00B4D8"], rot=0)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%d")
    ax.set_yscale("log"); ax.set_ylabel("Records (log scale)"); ax.set_title(title)
    fig.savefig(out_path); plt.close(fig)


def native_feature_importance(pipe) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity/split importance"
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).ravel(); kind = "|standardised coefficient|"
    else:
        return None
    imp = imp / imp.sum() if imp.sum() else imp
    return pd.DataFrame({"feature": names, "importance": imp, "kind": kind}).sort_values("importance", ascending=False)


def plot_feature_importances(pipes: dict, out_path, title="Native feature importance (normalised)"):
    _style()
    tables = {n: native_feature_importance(p) for n, p in pipes.items()}
    tables = {n: t for n, t in tables.items() if t is not None}
    n = len(tables); cols = min(3, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, t) in zip(axes, tables.items()):
        t = t.head(10).iloc[::-1]
        ax.barh(t["feature"], t["importance"], color="#1F6FEB")
        ax.set_title(f"{name}\n({t['kind'].iloc[0]})", fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold"); fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
