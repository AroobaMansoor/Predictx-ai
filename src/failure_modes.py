"""OPTIONAL advanced module: failure-mode analysis (multi-label).

The five AI4I failure modes (TWF, HDF, PWF, OSF, RNF) are NOT mutually exclusive:
some records carry several modes, some failures carry none and some RNF flags appear
without a machine failure. We therefore train one independent binary classifier per mode
(one-vs-rest, multi-label) instead of a single multi-class model.

These models are separate from the main binary failure predictor and the mode labels are
never used as inputs to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .data_loader import FAILURE_MODE_COLUMNS, FAILURE_MODE_NAMES, INPUT_COLUMNS
from .evaluation import compute_metrics, find_best_threshold
from .feature_engineering import FeatureEngineer
from .preprocessing import build_preprocessor
from imblearn.pipeline import Pipeline as ImbPipeline


def overlap_table(df: pd.DataFrame) -> pd.DataFrame:
    m = df[FAILURE_MODE_COLUMNS]
    co = m.T.dot(m)
    co.index = [FAILURE_MODE_NAMES[c] for c in co.index]
    co.columns = [FAILURE_MODE_NAMES[c] for c in co.columns]
    return co


def _pipe(seed: int):
    return ImbPipeline([
        ("features", FeatureEngineer("engineered")),
        ("preprocess", build_preprocessor("engineered", scale=False)),
        ("model", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                                 class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1)),
    ])


def train_failure_mode_models(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame]:
    models, rows = {}, []
    Xtr, Xva, Xte = train[INPUT_COLUMNS], val[INPUT_COLUMNS], test[INPUT_COLUMNS]
    for mode in FAILURE_MODE_COLUMNS:
        ytr, yva, yte = train[mode].astype(int), val[mode].astype(int), test[mode].astype(int)
        pipe = _pipe(seed).fit(Xtr, ytr)
        pva = pipe.predict_proba(Xva)[:, 1]
        thr = find_best_threshold(yva, pva) if yva.sum() > 0 else 0.5
        pte = pipe.predict_proba(Xte)[:, 1]
        m = compute_metrics(yte, pte, thr) if yte.sum() > 0 else {}
        rows.append({
            "mode": mode.upper(), "name": FAILURE_MODE_NAMES[mode],
            "train_positives": int(ytr.sum()), "val_positives": int(yva.sum()), "test_positives": int(yte.sum()),
            "threshold_from_val": thr,
            "test_pr_auc": float(average_precision_score(yte, pte)) if yte.sum() else np.nan,
            "test_roc_auc": float(roc_auc_score(yte, pte)) if 0 < yte.sum() < len(yte) else np.nan,
            "test_precision": m.get("precision", np.nan), "test_recall": m.get("recall", np.nan),
            "test_f1": m.get("f1", np.nan), "test_no_skill_pr_auc": float(yte.mean()),
        })
        models[mode] = {"pipeline": pipe, "threshold": thr}
    return models, pd.DataFrame(rows)


def predict_failure_modes(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for mode, obj in models.items():
        out[f"{mode}_probability"] = obj["pipeline"].predict_proba(X[INPUT_COLUMNS])[:, 1]
    return pd.DataFrame(out, index=X.index)
