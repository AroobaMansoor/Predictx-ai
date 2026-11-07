"""Splitting and preprocessing pipelines (fit on training data only)."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import FAILURE_MODE_COLUMNS, INPUT_COLUMNS, TARGET
from .feature_engineering import CATEGORICAL, feature_list


def split_data(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split (default 60 / 20 / 20)."""
    seed = cfg["seed"]
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    strat = df[TARGET] if cfg["split"].get("stratify", True) else None
    train_val, test = train_test_split(df, test_size=test_size, stratify=strat, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    strat_tv = train_val[TARGET] if strat is not None else None
    train, val = train_test_split(train_val, test_size=rel_val, stratify=strat_tv, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and binary target. IDs and failure-mode labels are excluded."""
    X = df[INPUT_COLUMNS].copy()
    y = df[TARGET].astype(int).copy()
    forbidden = set(FAILURE_MODE_COLUMNS) | {TARGET, "udi", "product_id"}
    assert not forbidden & set(X.columns), "Leakage: label/ID columns in features"
    return X, y


def build_preprocessor(feature_set: str = "engineered", scale: bool = True,
                       drop: list[str] | None = None, encode_categorical: bool = True):
    """ColumnTransformer: impute + (scale) numerics, one-hot encode product type.

    ``encode_categorical=False`` returns a passthrough for models with native
    categorical support (CatBoost).
    """
    cols = feature_list(feature_set, drop)
    num_cols = [c for c in cols if c not in CATEGORICAL]
    cat_cols = [c for c in cols if c in CATEGORICAL]

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num_cols)]
    if cat_cols:
        if encode_categorical:
            cat_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        else:
            cat_pipe = "passthrough"
        transformers.append(("cat", cat_pipe, cat_cols))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct
