"""Shared helpers: configuration, logging, reproducibility, JSON I/O."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

try:  # PyYAML is optional; config.json is used as a fallback
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PredictXError(Exception):
    """Base exception for PredictX AI."""


class DataValidationError(PredictXError):
    """Raised when the dataset does not match the expected schema."""


class InputValidationError(PredictXError):
    """Raised when prediction inputs are invalid."""


class ModelNotFoundError(PredictXError):
    """Raised when trained artifacts are missing."""


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (defaults to the project root)."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if path.suffix in {".yaml", ".yml"} and yaml is None:
        # PyYAML not installed (e.g. very new Python versions): use the JSON copy
        path = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh) if path.suffix == ".json" else yaml.safe_load(fh)
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dirs(cfg: dict) -> None:
    for key in ("processed_dir", "models_dir", "figures_dir", "reports_dir", "predictions_dir", "logs_dir"):
        resolve_path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Fix every random source we use."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "predictx", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve() for h in logger.handlers):
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.propagate = False
    return logger


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@contextmanager
def timer(logger: logging.Logger | None = None, label: str = "block"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if logger:
        logger.info("%s finished in %.2fs", label, elapsed)
