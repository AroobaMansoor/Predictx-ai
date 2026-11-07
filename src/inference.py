"""Inference: input validation, single and batch prediction with SHAP and recommendations."""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_loader import FAILURE_MODE_NAMES, INPUT_COLUMNS, REVERSE_COLUMN_MAP, standardize_columns
from .explainability import ShapExplainer, group_onehot, local_contributions
from .recommendations import generate_recommendation
from .utils import InputValidationError, ModelNotFoundError, load_config, resolve_path

NUMERIC_INPUTS = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
# friendly aliases accepted in batch CSVs and API dictionaries
ALIASES = {
    "machine_type": "type", "product_type": "type",
    "air_temperature": "air_temp_k", "air_temp": "air_temp_k",
    "process_temperature": "process_temp_k", "process_temp": "process_temp_k",
    "rotational_speed": "rotational_speed_rpm", "rpm": "rotational_speed_rpm",
    "torque": "torque_nm", "tool_wear": "tool_wear_min",
}
DEFAULT_MODEL_FILE = "predictx_best_model.joblib"
FAILURE_MODE_FILE = "failure_mode_models.joblib"


def risk_category(p: float, low_max: float, high_min: float) -> str:
    if p >= high_min:
        return "High"
    if p < low_max:
        return "Low"
    return "Medium"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    lower = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=lower)
    return df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})


def validate_input(inputs: dict, cfg: dict, training_ranges: dict | None = None) -> tuple[dict, list[str]]:
    """Validate one machine record. Returns (clean_inputs, warnings) or raises InputValidationError."""
    if not isinstance(inputs, dict):
        raise InputValidationError("Input must be a dictionary")
    data = {ALIASES.get(k, k): v for k, v in normalize_columns(pd.DataFrame([inputs])).iloc[0].items()}
    missing = [c for c in INPUT_COLUMNS if c not in data or data[c] is None or
               (isinstance(data[c], float) and math.isnan(data[c]))]
    if missing:
        raise InputValidationError(f"Missing required inputs: {missing}")

    clean, warnings = {}, []
    t = str(data["type"]).strip().upper()
    if t not in cfg["validation"]["allowed_types"]:
        raise InputValidationError(f"Machine type must be one of {cfg['validation']['allowed_types']}, got '{data['type']}'")
    clean["type"] = t
    for col in NUMERIC_INPUTS:
        try:
            val = float(data[col])
        except (TypeError, ValueError):
            raise InputValidationError(f"'{col}' must be numeric, got {data[col]!r}") from None
        if not math.isfinite(val):
            raise InputValidationError(f"'{col}' must be a finite number")
        lo, hi = cfg["validation"]["hard_limits"][col]
        if not lo <= val <= hi:
            raise InputValidationError(f"'{col}' = {val} is outside the allowed range [{lo}, {hi}]")
        clean[col] = val
        if training_ranges and col in training_ranges:
            tlo, thi = training_ranges[col]
            if not tlo <= val <= thi:
                warnings.append(f"{REVERSE_COLUMN_MAP.get(col, col)} = {val:g} is outside the training range "
                                f"[{tlo:g}, {thi:g}]; the prediction is an extrapolation and less reliable.")
    if clean["process_temp_k"] < clean["air_temp_k"]:
        warnings.append("Process temperature is below air temperature, which never occurs in the training data.")
    return clean, warnings


class PredictXPredictor:
    """Loads the saved best-model artifact and serves predictions."""

    def __init__(self, model_path: str | Path | None = None, cfg: dict | None = None, load_failure_modes: bool = True):
        self.cfg = cfg or load_config()
        models_dir = resolve_path(self.cfg["paths"]["models_dir"])
        path = Path(model_path) if model_path else models_dir / DEFAULT_MODEL_FILE
        if not path.exists():
            raise ModelNotFoundError(f"Trained model not found at {path}. Run `python train.py` first.")
        self.artifact = joblib.load(path)
        required = {"model", "explain_pipeline", "threshold", "background", "metadata"}
        if not required.issubset(self.artifact):
            raise ModelNotFoundError(f"Artifact at {path} is incomplete; missing {required - set(self.artifact)}")
        self.model = self.artifact["model"]
        self.threshold = float(self.artifact["threshold"])
        self.metadata = self.artifact["metadata"]
        self.training_ranges = self.artifact.get("training_ranges", {})
        risk = self.cfg["risk"]
        self.low_max = float(risk["low_max"])
        self.high_min = self.threshold if risk.get("high_min", "auto") == "auto" else float(risk["high_min"])
        self.high_min = max(self.high_min, self.low_max)
        self._explainer = None
        self.failure_mode_models = None
        fm_path = models_dir / FAILURE_MODE_FILE
        if load_failure_modes and fm_path.exists():
            self.failure_mode_models = joblib.load(fm_path)

    @property
    def explainer(self) -> ShapExplainer:
        if self._explainer is None:
            self._explainer = ShapExplainer(self.artifact["explain_pipeline"], self.artifact["background"],
                                            seed=self.cfg["seed"],
                                            background_size=self.cfg["explainability"]["background_size"])
        return self._explainer

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[INPUT_COLUMNS])[:, 1]

    def predict(self, inputs: dict, explain: bool = True, top_k: int | None = None) -> dict:
        clean, warnings = validate_input(inputs, self.cfg, self.training_ranges)
        X = pd.DataFrame([clean])[INPUT_COLUMNS]
        p = float(self.predict_proba(X)[0])
        cat = risk_category(p, self.low_max, self.high_min)
        top_k = top_k or self.cfg["explainability"]["top_k"]
        top, shap_payload = [], None
        if explain:
            expl = group_onehot(self.explainer.explain(X), raw_X=X)
            top = local_contributions(expl[0], X.iloc[0], top_k=top_k)
            shap_payload = {"values": expl.values[0].tolist(), "base_value": float(expl.base_values[0]),
                            "feature_names": list(expl.feature_names), "data": expl.data[0].tolist(),
                            "output_space": self.explainer.output_space}
        result = {
            "inputs": clean,
            "predicted_class": "Failure" if p >= self.threshold else "No Failure",
            "failure_probability": p,
            "risk_category": cat,
            "decision_threshold": self.threshold,
            "risk_thresholds": {"low_max": self.low_max, "high_min": self.high_min},
            "top_features": top,
            "shap": shap_payload,
            "recommendation": generate_recommendation(clean, p, cat, top, self.cfg["recommendations"]),
            "warnings": warnings,
            "model_name": self.metadata.get("model_display_name"),
        }
        if self.failure_mode_models:
            result["failure_modes"] = {
                FAILURE_MODE_NAMES[m]: float(o["pipeline"].predict_proba(X)[:, 1][0])
                for m, o in self.failure_mode_models.items()}
        return result

    def predict_batch(self, df: pd.DataFrame, explain: bool = True, top_k: int = 3) -> pd.DataFrame:
        """Validate every row; invalid rows get an error message instead of a prediction."""
        if df is None or len(df) == 0:
            raise InputValidationError("The uploaded file contains no rows")
        norm = normalize_columns(df.copy())
        missing = [c for c in INPUT_COLUMNS if c not in norm.columns]
        if missing:
            raise InputValidationError(
                f"CSV is missing required columns: {[REVERSE_COLUMN_MAP.get(c, c) for c in missing]}. "
                "Use the original UCI headers (e.g. 'Air temperature [K]') or snake_case names (e.g. 'air_temp_k').")
        clean_rows, errors, warns = [], [], []
        for _, row in norm[INPUT_COLUMNS].iterrows():
            try:
                c, w = validate_input(row.to_dict(), self.cfg, self.training_ranges)
                clean_rows.append(c); errors.append(""); warns.append(" | ".join(w))
            except InputValidationError as e:
                clean_rows.append(None); errors.append(str(e)); warns.append("")
        out = df.copy().reset_index(drop=True)
        valid_idx = [i for i, c in enumerate(clean_rows) if c is not None]
        out["failure_probability"] = np.nan
        out["predicted_class"] = ""
        out["risk_category"] = ""
        out["top_risk_drivers"] = ""
        out["maintenance_priority"] = ""
        out["recommended_actions"] = ""
        out["validation_warning"] = warns
        out["validation_error"] = errors
        if not valid_idx:
            return out
        X = pd.DataFrame([clean_rows[i] for i in valid_idx])[INPUT_COLUMNS]
        p = self.predict_proba(X)
        out.loc[valid_idx, "failure_probability"] = p
        out.loc[valid_idx, "predicted_class"] = np.where(p >= self.threshold, "Failure", "No Failure")
        cats = [risk_category(float(v), self.low_max, self.high_min) for v in p]
        out.loc[valid_idx, "risk_category"] = cats
        tops = [[] for _ in valid_idx]
        if explain:
            expl = group_onehot(self.explainer.explain(X), raw_X=X)
            tops = [local_contributions(expl[j], X.iloc[j], top_k=top_k) for j in range(len(X))]
            out.loc[valid_idx, "top_risk_drivers"] = [
                "; ".join(f"{t['label']} ({'+' if t['shap_value'] > 0 else '-'}{abs(t['shap_value']):.2f})" for t in tt)
                for tt in tops]
        recs = [generate_recommendation(X.iloc[j].to_dict(), float(p[j]), cats[j], tops[j], self.cfg["recommendations"])
                for j in range(len(X))]
        out.loc[valid_idx, "maintenance_priority"] = [r["priority"] for r in recs]
        out.loc[valid_idx, "recommended_actions"] = [" ".join(r["actions"]) for r in recs]
        return out
