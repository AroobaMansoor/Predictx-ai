"""Loading, downloading and validating the AI4I 2020 Predictive Maintenance dataset.

The raw CSV uses column names such as ``Air temperature [K]``. Internally we use
snake_case names; :data:`COLUMN_MAP` defines the mapping.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .utils import DataValidationError, get_logger, resolve_path

logger = get_logger("predictx.data")

COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}
REVERSE_COLUMN_MAP = {v: k for k, v in COLUMN_MAP.items()}

INPUT_COLUMNS = ["type", "air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
FAILURE_MODE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
FAILURE_MODE_NAMES = {
    "twf": "Tool Wear Failure",
    "hdf": "Heat Dissipation Failure",
    "pwf": "Power Failure",
    "osf": "Overstrain Failure",
    "rnf": "Random Failure",
}
TARGET = "machine_failure"
EXPECTED_ROWS = 10_000
VALID_TYPES = {"L", "M", "H"}


def download_dataset(url: str, dest: Path) -> Path:
    """Download the official UCI ZIP and extract ai4i2020.csv to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading AI4I 2020 dataset from %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed official URL
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise DataValidationError("Downloaded archive does not contain a CSV file")
        with zf.open(csv_names[0]) as fh, open(dest, "wb") as out:
            out.write(fh.read())
    logger.info("Saved dataset to %s", dest)
    return dest


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw UCI column names to snake_case (already-snake names are kept)."""
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})


def validate_dataset(df: pd.DataFrame, expect_full: bool = True) -> dict:
    """Validate schema, types, value domains and label consistency.

    Returns a report dict. Raises :class:`DataValidationError` on hard failures.
    """
    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing columns: {missing}")

    numeric = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min",
               TARGET] + FAILURE_MODE_COLUMNS
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric, got {df[col].dtype}")

    bad_types = set(df["type"].dropna().unique()) - VALID_TYPES
    if bad_types:
        raise DataValidationError(f"Unexpected product types: {bad_types}")

    for col in [TARGET] + FAILURE_MODE_COLUMNS:
        if not set(df[col].dropna().unique()).issubset({0, 1}):
            raise DataValidationError(f"Label column '{col}' must be binary 0/1")

    for col in ["rotational_speed_rpm", "torque_nm", "tool_wear_min"]:
        if (df[col] < 0).any():
            raise DataValidationError(f"Negative values found in '{col}'")

    mode_sum = df[FAILURE_MODE_COLUMNS].sum(axis=1)
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows_all_columns": int(df.duplicated().sum()),
        "duplicate_rows_feature_columns": int(df.duplicated(subset=INPUT_COLUMNS).sum()),
        "duplicate_ids": int(df["udi"].duplicated().sum()),
        "n_failures": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()),
        "failures_without_mode_label": int(((df[TARGET] == 1) & (mode_sum == 0)).sum()),
        "mode_label_without_failure": int(((df[TARGET] == 0) & (mode_sum > 0)).sum()),
        "rows_with_multiple_modes": int((mode_sum > 1).sum()),
    }
    if expect_full and report["n_rows"] != EXPECTED_ROWS:
        logger.warning("Expected %d rows, found %d", EXPECTED_ROWS, report["n_rows"])
    return report


def load_dataset(cfg: dict, validate: bool = True, allow_download: bool = True) -> pd.DataFrame:
    """Load the raw AI4I CSV with standardized column names."""
    path = resolve_path(cfg["paths"]["raw_data"])
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Dataset not found at {path}")
        download_dataset(cfg["dataset"]["uci_zip_url"], path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = standardize_columns(df)
    df["type"] = df["type"].astype(str).str.strip()
    if validate:
        report = validate_dataset(df)
        logger.info("Dataset validated: %d rows, %d failures (%.2f%%), %d missing values",
                    report["n_rows"], report["n_failures"], 100 * report["failure_rate"], report["total_missing"])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without a target; numeric NaNs are imputed later inside the pipeline.

    The official dataset has no missing values, but the pipeline stays robust to them.
    """
    before = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) != before:
        logger.warning("Dropped %d rows with missing target", before - len(df))
    return df
