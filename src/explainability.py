"""SHAP explanations for the trained pipelines.

IMPORTANT: SHAP values describe how the *model* uses each input to reach its score.
They are statistical attributions, not proven physical causes of equipment failure.

Output space: tree boosters (XGBoost, LightGBM, CatBoost) and Logistic Regression are
explained in log-odds; Random Forest and the MLP in probability space.
If the deployed model is calibrated, SHAP explains the underlying (uncalibrated)
pipeline. Calibration is monotonic, so the ranking and direction of contributions stay
the same.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .train_models import model_family

TYPE_CODES = {"L": 0, "M": 1, "H": 2}
PRETTY = {
    "type": "Product type", "air_temp_k": "Air temperature [K]", "process_temp_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]", "torque_nm": "Torque [Nm]", "tool_wear_min": "Tool wear [min]",
    "temp_diff_k": "Temp. difference [K]", "power_w": "Power [W]", "wear_torque_min_nm": "Wear x Torque [min·Nm]",
    "temp_diff_x_torque": "Temp. diff x Torque [K·Nm]",
}


def transform_inputs(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering + preprocessing exactly as the model sees it (no resampling)."""
    Xf = pipe.named_steps["features"].transform(X)
    return pipe.named_steps["preprocess"].transform(Xf)


class ShapExplainer:
    """Model-aware SHAP wrapper around a fitted PredictX pipeline."""

    def __init__(self, pipe, background: pd.DataFrame, seed: int = 42, background_size: int = 200):
        self.pipe = pipe
        self.family = model_family(pipe)
        self.model = pipe.named_steps["model"]
        bg = background.sample(min(background_size, len(background)), random_state=seed)
        self.background_t = transform_inputs(pipe, bg)
        self.feature_names = list(self.background_t.columns)
        self.output_space = "log-odds"
        if self.family in {"random_forest", "neural_network"}:
            self.output_space = "probability"
        if self.family in {"random_forest", "xgboost", "lightgbm"}:
            self._explainer = shap.TreeExplainer(self.model)
        elif self.family == "logistic_regression":
            self._explainer = shap.LinearExplainer(self.model, self.background_t)
        elif self.family == "catboost":
            self._explainer = None  # CatBoost's native, exact TreeSHAP handles categoricals
        else:
            bg_small = shap.sample(self.background_t, min(50, len(self.background_t)), random_state=seed)
            f = lambda a: self.model.predict_proba(pd.DataFrame(a, columns=self.feature_names))[:, 1]  # noqa: E731
            self._explainer = shap.KernelExplainer(f, bg_small)

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        Xt = transform_inputs(self.pipe, X)
        if self.family == "catboost":
            from catboost import Pool
            cat_idx = [i for i, c in enumerate(Xt.columns) if c == "type"]
            sv = self.model.get_feature_importance(Pool(Xt, cat_features=cat_idx), type="ShapValues")
            values, base = sv[:, :-1], sv[:, -1]
        elif self.family == "neural_network":
            values = np.asarray(self._explainer.shap_values(Xt, nsamples=200, silent=True))
            base = np.full(len(Xt), float(np.ravel(self._explainer.expected_value)[0]))
        else:
            res = self._explainer(Xt)
            values, base = res.values, res.base_values
            if values.ndim == 3:  # (n, features, classes) -> positive class
                values, base = values[:, :, 1], np.asarray(base)[:, 1]
        base = np.broadcast_to(np.asarray(base, dtype=float).ravel(), (len(Xt),)).copy() \
            if np.size(base) in (1, len(Xt)) else np.asarray(base, dtype=float)
        data = Xt.copy()
        if "type" in data.columns:  # CatBoost: keep a numeric code for plotting colour scales
            data["type"] = data["type"].map(TYPE_CODES).astype(float)
        return shap.Explanation(values=np.asarray(values, dtype=float), base_values=base,
                                data=data.to_numpy(dtype=float), feature_names=self.feature_names)


def group_onehot(expl: shap.Explanation, raw_X: pd.DataFrame | None = None) -> shap.Explanation:
    """Merge one-hot columns ``type_L/M/H`` into a single 'type' attribution."""
    names = list(expl.feature_names)
    type_cols = [i for i, n in enumerate(names) if n.startswith("type_")]
    if not type_cols:
        return expl
    keep = [i for i in range(len(names)) if i not in type_cols]
    values = np.column_stack([expl.values[:, keep], expl.values[:, type_cols].sum(axis=1)])
    if raw_X is not None:
        type_data = raw_X["type"].map(TYPE_CODES).to_numpy(dtype=float)
    else:
        sub = expl.data[:, type_cols]
        order = [names[i].split("_", 1)[1] for i in type_cols]
        type_data = np.array([TYPE_CODES.get(order[int(np.argmax(r))], np.nan) for r in sub])
    data = np.column_stack([expl.data[:, keep], type_data])
    return shap.Explanation(values=values, base_values=expl.base_values, data=data,
                            feature_names=[names[i] for i in keep] + ["type"])


def global_importance(expl: shap.Explanation) -> pd.DataFrame:
    imp = np.abs(expl.values).mean(axis=0)
    df = pd.DataFrame({"feature": expl.feature_names, "mean_abs_shap": imp,
                       "mean_shap": expl.values.mean(axis=0)})
    df["share_pct"] = 100 * df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    df["label"] = df["feature"].map(lambda f: PRETTY.get(f, f))
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_contributions(expl_row: shap.Explanation, raw_row: pd.Series, top_k: int = 5) -> list[dict]:
    """Top-k contributions of a single (grouped) explanation row."""
    vals = np.asarray(expl_row.values).ravel()
    feats = list(expl_row.feature_names)
    order = np.argsort(-np.abs(vals))[:top_k]
    out = []
    for i in order:
        f = feats[i]
        value = raw_row.get(f, expl_row.data[i]) if f in raw_row.index else expl_row.data[i]
        out.append({"feature": f, "label": PRETTY.get(f, f), "value": value if isinstance(value, str) else float(value),
                    "shap_value": float(vals[i]),
                    "direction": "increases risk" if vals[i] > 0 else "decreases risk"})
    return out


def _pretty(expl: shap.Explanation) -> shap.Explanation:
    e = shap.Explanation(values=expl.values, base_values=expl.base_values, data=expl.data,
                         feature_names=[PRETTY.get(f, f) for f in expl.feature_names])
    return e


def save_summary_plot(expl, out_path, title):
    plt.figure()
    shap.plots.beeswarm(_pretty(expl), max_display=12, show=False)
    plt.title(title, fontweight="bold"); plt.gcf().set_size_inches(9, 6)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close("all")


def save_bar_plot(expl, out_path, title):
    plt.figure()
    shap.plots.bar(_pretty(expl), max_display=12, show=False)
    plt.title(title, fontweight="bold"); plt.gcf().set_size_inches(9, 6)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close("all")


def save_waterfall_plot(expl_row, out_path, title):
    plt.figure()
    shap.plots.waterfall(_pretty(expl_row), max_display=10, show=False)
    plt.title(title, fontweight="bold"); plt.gcf().set_size_inches(9, 6)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close("all")


def save_dependence_plots(expl, features: list[str], out_path, title):
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.3))
    axes = np.atleast_1d(axes)
    pe = _pretty(expl)
    for ax, f in zip(axes, features):
        idx = list(expl.feature_names).index(f)
        shap.plots.scatter(pe[:, idx], color=pe, ax=ax, show=False)
        ax.set_title(PRETTY.get(f, f), fontsize=10)
    fig.suptitle(title, fontweight="bold"); fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close("all")
