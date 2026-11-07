"""Physically-motivated engineered features.

All features are derived ONLY from the six input variables — never from the target
or failure-mode labels — so they introduce no target leakage.

| Feature              | Formula                                   | Unit      | Rationale                                  |
|----------------------|-------------------------------------------|-----------|--------------------------------------------|
| temp_diff_k          | process_temp_k - air_temp_k               | K         | Heat-dissipation capacity                  |
| power_w              | torque_nm * rotational_speed_rpm * 2π/60  | W         | Mechanical power  P = τ·ω                  |
| wear_torque_min_nm   | tool_wear_min * torque_nm                 | min·Nm    | Cumulative strain on the tool (overstrain) |
| temp_diff_x_torque   | temp_diff_k * torque_nm                   | K·Nm      | Thermal × mechanical load interaction      |

Assumptions: temperatures are in Kelvin, speed in revolutions per minute, torque in
newton-metres, tool wear in minutes (as documented by UCI). ω = rpm·2π/60 rad/s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_NUMERIC = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
CATEGORICAL = ["type"]
ENGINEERED = ["temp_diff_k", "power_w", "wear_torque_min_nm", "temp_diff_x_torque"]

FEATURE_DOCS = {
    "temp_diff_k": {"formula": "process_temp_k - air_temp_k", "unit": "K"},
    "power_w": {"formula": "torque_nm * rotational_speed_rpm * 2*pi/60", "unit": "W"},
    "wear_torque_min_nm": {"formula": "tool_wear_min * torque_nm", "unit": "min*Nm"},
    "temp_diff_x_torque": {"formula": "(process_temp_k - air_temp_k) * torque_nm", "unit": "K*Nm"},
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered columns appended."""
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2.0 * np.pi / 60.0
    out["wear_torque_min_nm"] = out["tool_wear_min"] * out["torque_nm"]
    out["temp_diff_x_torque"] = out["temp_diff_k"] * out["torque_nm"]
    return out


def feature_list(feature_set: str, drop: list[str] | None = None) -> list[str]:
    """Ordered model input columns for a feature set ('baseline' or 'engineered')."""
    if feature_set not in {"baseline", "engineered"}:
        raise ValueError(f"Unknown feature_set '{feature_set}'")
    cols = CATEGORICAL + BASE_NUMERIC + (ENGINEERED if feature_set == "engineered" else [])
    drop = set(drop or [])
    return [c for c in cols if c not in drop]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer: selects inputs and adds engineered features.

    Being stateless it cannot leak information from validation/test data.
    """

    def __init__(self, feature_set: str = "engineered", drop: list[str] | None = None):
        self.feature_set = feature_set
        self.drop = drop

    def fit(self, X, y=None):
        self.columns_ = feature_list(self.feature_set, self.drop)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        missing = [c for c in CATEGORICAL + BASE_NUMERIC if c not in X.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        X = X[CATEGORICAL + BASE_NUMERIC].copy()
        if self.feature_set == "engineered":
            X = add_engineered_features(X)
        cols = feature_list(self.feature_set, self.drop)
        return X[cols]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(feature_list(self.feature_set, self.drop), dtype=object)
