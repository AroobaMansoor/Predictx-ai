"""Model registry, pipelines, hyperparameter search spaces and training helpers.

Every model is wrapped in an ``imblearn`` Pipeline:

    features (FeatureEngineer) -> preprocess (ColumnTransformer) -> resample -> model

so that feature engineering, scaling/encoding and any resampling (SMOTE / undersampling)
are fitted inside each training fold only. Validation and test data are never resampled
(imblearn samplers are skipped at predict time).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
try:  # CatBoost is optional (no wheels for some very new Python versions)
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .feature_engineering import FeatureEngineer
from .preprocessing import build_preprocessor

warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

class _CatBoostBase:
    pass


class CatBoostCategorical(CatBoostClassifier if CATBOOST_AVAILABLE else _CatBoostBase):
    """CatBoost that detects string/categorical columns at fit time.

    Passing ``cat_features`` to the constructor breaks ``sklearn.base.clone``; detecting
    them in ``fit`` keeps CatBoost's native categorical handling and full sklearn
    compatibility (CV, RandomizedSearchCV, CalibratedClassifierCV).
    """

    def fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame) and "cat_features" not in kwargs:
            kwargs["cat_features"] = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        return super().fit(X, y, **kwargs)


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "neural_network": "Neural Network (MLP)",
}
# models whose inputs should be standardised
NEEDS_SCALING = {"logistic_regression", "neural_network"}
# models that use native categorical handling instead of one-hot encoding
NATIVE_CATEGORICAL = {"catboost"}


def make_estimator(name: str, seed: int):
    """Library-default estimator (used as the untuned baseline in Experiment 4)."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if name == "xgboost":
        return XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss", tree_method="hist")
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1)
    if name == "catboost":
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost is not installed")
        return CatBoostCategorical(random_seed=seed, verbose=0, thread_count=1, allow_writing_files=False)
    if name == "neural_network":
        return MLPClassifier(max_iter=1000, random_state=seed)
    raise ValueError(f"Unknown model '{name}'")


def build_pipeline(name: str, seed: int, feature_set: str = "engineered", drop: list[str] | None = None,
                   imbalance: str = "none", pos_weight: float = 1.0) -> ImbPipeline:
    pre = build_preprocessor(feature_set=feature_set, scale=name in NEEDS_SCALING, drop=drop,
                             encode_categorical=name not in NATIVE_CATEGORICAL)
    pipe = ImbPipeline([
        ("features", FeatureEngineer(feature_set=feature_set, drop=drop)),
        ("preprocess", pre),
        ("resample", "passthrough"),
        ("model", make_estimator(name, seed)),
    ])
    return set_imbalance_strategy(pipe, name, imbalance, seed, pos_weight)


def supported_imbalance_strategies(name: str) -> list[str]:
    base = ["none", "smote", "random_undersample"]
    if name == "catboost":  # SMOTE cannot interpolate the raw categorical column
        base = ["none", "random_undersample"]
    if name != "neural_network":  # MLPClassifier has no class_weight parameter
        base.insert(1, "class_weight")
    return base


def set_imbalance_strategy(pipe: ImbPipeline, name: str, strategy: str, seed: int, pos_weight: float) -> ImbPipeline:
    """Configure one imbalance-handling strategy on a pipeline (returns the same object)."""
    if strategy not in supported_imbalance_strategies(name):
        raise ValueError(f"Strategy '{strategy}' not supported for {name}")
    # reset any class weighting
    weight_param = {"logistic_regression": ("class_weight", None, "balanced"),
                    "random_forest": ("class_weight", None, "balanced"),
                    "xgboost": ("scale_pos_weight", 1.0, pos_weight),
                    "lightgbm": ("class_weight", None, "balanced"),
                    "catboost": ("auto_class_weights", "None", "Balanced")}.get(name)
    if weight_param:
        param, off, on = weight_param
        pipe.set_params(**{f"model__{param}": on if strategy == "class_weight" else off})
    if strategy == "smote":
        pipe.set_params(resample=SMOTE(random_state=seed, k_neighbors=5))
    elif strategy == "random_undersample":
        pipe.set_params(resample=RandomUnderSampler(random_state=seed))
    else:
        pipe.set_params(resample="passthrough")
    return pipe


def param_distributions(name: str, seed: int, pos_weight: float) -> dict:
    """Randomized-search spaces. Class weighting / resampling is part of the search."""
    if name == "logistic_regression":
        return {"model__C": loguniform(1e-3, 1e2), "model__class_weight": [None, "balanced"]}
    if name == "random_forest":
        return {"model__n_estimators": randint(200, 601), "model__max_depth": [None, 6, 10, 16, 24],
                "model__min_samples_split": randint(2, 11), "model__min_samples_leaf": randint(1, 6),
                "model__max_features": ["sqrt", 0.5, 0.8],
                "model__class_weight": [None, "balanced", "balanced_subsample"]}
    if name == "xgboost":
        return {"model__n_estimators": randint(200, 801), "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(3, 9), "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4), "model__min_child_weight": randint(1, 8),
                "model__gamma": uniform(0, 2), "model__reg_lambda": loguniform(1e-2, 10),
                "model__scale_pos_weight": [1.0, float(np.sqrt(pos_weight)), float(pos_weight)]}
    if name == "lightgbm":
        return {"model__n_estimators": randint(200, 801), "model__learning_rate": loguniform(0.01, 0.2),
                "model__num_leaves": randint(8, 64), "model__max_depth": [-1, 4, 6, 8],
                "model__min_child_samples": randint(5, 50), "model__subsample": uniform(0.6, 0.4),
                "model__subsample_freq": [1], "model__colsample_bytree": uniform(0.6, 0.4),
                "model__reg_lambda": loguniform(1e-3, 10), "model__class_weight": [None, "balanced"]}
    if name == "catboost":
        return {"model__iterations": randint(300, 901), "model__learning_rate": loguniform(0.02, 0.2),
                "model__depth": randint(4, 8), "model__l2_leaf_reg": loguniform(1, 10),
                "model__auto_class_weights": ["None", "Balanced", "SqrtBalanced"]}
    if name == "neural_network":
        return {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                "model__alpha": loguniform(1e-5, 1e-2), "model__learning_rate_init": loguniform(1e-4, 1e-2),
                "model__early_stopping": [True], "model__n_iter_no_change": [20],
                "resample": ["passthrough", SMOTE(random_state=seed)]}
    raise ValueError(name)


def available_models(names: list[str]) -> list[str]:
    """Drop models whose library is not installed."""
    return [n for n in names if n != "catboost" or CATBOOST_AVAILABLE]


def cv_splitter(cfg: dict) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cfg["training"]["cv_folds"], shuffle=True, random_state=cfg["seed"])


def tune_model(name: str, X: pd.DataFrame, y: pd.Series, cfg: dict, feature_set: str, logger=None):
    """RandomizedSearchCV with stratified K-fold on the TRAINING set only."""
    seed = cfg["seed"]
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    pipe = build_pipeline(name, seed, feature_set=feature_set, pos_weight=pos_weight)
    search = RandomizedSearchCV(
        pipe, param_distributions(name, seed, pos_weight), n_iter=cfg["training"]["search_iterations"],
        scoring=cfg["training"]["scoring"], cv=cv_splitter(cfg), n_jobs=cfg["training"]["n_jobs"],
        random_state=seed, refit=True, return_train_score=False, error_score="raise")
    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0
    if logger:
        logger.info("  %-22s best CV %s = %.4f  (%.1fs)", MODEL_DISPLAY_NAMES[name], cfg["training"]["scoring"],
                    search.best_score_, elapsed)
    cv_res = pd.DataFrame(search.cv_results_)
    info = {"best_cv_score": float(search.best_score_),
            "best_cv_std": float(cv_res.loc[search.best_index_, "std_test_score"]),
            "best_params": {k: (str(v) if not isinstance(v, (int, float, str, type(None))) else v)
                            for k, v in search.best_params_.items()},
            "search_time_s": elapsed, "n_candidates": len(cv_res)}
    return search.best_estimator_, info, cv_res


def cross_validate_pipeline(pipe, X, y, cfg: dict) -> tuple[float, float]:
    scores = cross_val_score(clone(pipe), X, y, cv=cv_splitter(cfg), scoring=cfg["training"]["scoring"],
                             n_jobs=cfg["training"]["n_jobs"])
    return float(scores.mean()), float(scores.std())


def model_family(pipe) -> str:
    """Infer registry name from a fitted pipeline (or CalibratedClassifierCV wrapper)."""
    est = pipe.named_steps["model"] if hasattr(pipe, "named_steps") else pipe
    mapping = {LogisticRegression: "logistic_regression", RandomForestClassifier: "random_forest",
               XGBClassifier: "xgboost", LGBMClassifier: "lightgbm", MLPClassifier: "neural_network"}
    if CATBOOST_AVAILABLE:
        mapping[CatBoostClassifier] = "catboost"
    for cls, n in mapping.items():
        if isinstance(est, cls):
            return n
    return type(est).__name__
