"""PredictX AI — full training & experiment pipeline.

Usage:
    python train.py                # full run (all experiments)
    python train.py --quick        # fast smoke run (few search iterations)

Protocol (no test-set leakage):
  * stratified 60/20/20 train / validation / test split (seeded)
  * all fitting, CV and hyperparameter search use TRAIN only
  * model selection, calibration choice and decision threshold use VALIDATION only
  * TEST is used exactly once at the end for reporting
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src import eda as eda_mod
from src.data_loader import FAILURE_MODE_COLUMNS, FAILURE_MODE_NAMES, INPUT_COLUMNS, TARGET, handle_missing_values, load_dataset
from src.evaluation import (classification_report_dict, compute_metrics, find_best_threshold, measure_inference_time,
                            plot_calibration, plot_class_distribution, plot_confusion_matrices,
                            plot_feature_importances, plot_model_comparison, plot_pr_curves, plot_roc_curves)
from src.explainability import (ShapExplainer, global_importance, group_onehot, save_bar_plot,
                                save_dependence_plots, save_summary_plot, save_waterfall_plot)
from src.feature_engineering import FEATURE_DOCS, feature_list
from src.failure_modes import overlap_table, train_failure_mode_models
from src.preprocessing import get_X_y, split_data
from src.train_models import (MODEL_DISPLAY_NAMES, available_models, build_pipeline, cross_validate_pipeline,
                              set_imbalance_strategy, supported_imbalance_strategies, tune_model, cv_splitter)
from src.utils import ensure_dirs, get_logger, load_config, resolve_path, save_json, set_seed

warnings.filterwarnings("ignore")
KEY_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate",
               "false_negative_rate", "brier", "ece", "mcc"]


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def library_versions() -> dict:
    import imblearn, lightgbm, shap, sklearn, xgboost  # noqa: E401
    try:
        import catboost
        cb = catboost.__version__
    except ImportError:
        cb = "not installed"
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "catboost": cb, "imbalanced-learn": imblearn.__version__, "shap": shap.__version__}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train PredictX AI models and run all experiments")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="few search iterations, for smoke testing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.quick:
        cfg["training"]["search_iterations"] = 2
        cfg["training"]["cv_folds"] = 3
    seed = cfg["seed"]
    set_seed(seed)
    ensure_dirs(cfg)
    P = {k: resolve_path(v) for k, v in cfg["paths"].items()}
    (P["reports_dir"] / "tuning").mkdir(parents=True, exist_ok=True)
    (P["models_dir"] / "all_models").mkdir(parents=True, exist_ok=True)
    log = get_logger("predictx.train", P["logs_dir"] / "train.log")
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("PredictX AI training run started (quick=%s, seed=%d)", args.quick, seed)
    log.info("Library versions: %s", library_versions())
    figs, reps = P["figures_dir"], P["reports_dir"]

    # ------------------------------------------------------------ Phase 1-2: data + EDA
    df = handle_missing_values(load_dataset(cfg))
    eda_summary = eda_mod.compute_eda_summary(df)
    save_json(eda_summary, reps / "eda_summary.json")
    eda_mod.generate_eda_figures(df, figs)
    for o in eda_summary["observations"]:
        log.info("EDA: %s", o)

    train, val, test = split_data(df, cfg)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(P["processed_dir"] / f"{name}.csv", index=False)
    split_info = {n: {"rows": len(d), "failures": int(d[TARGET].sum()), "failure_rate": float(d[TARGET].mean())}
                  for n, d in [("train", train), ("validation", val), ("test", test)]}
    save_json(split_info, reps / "split_summary.json")
    log.info("Split: %s", split_info)
    X_tr, y_tr = get_X_y(train)
    X_va, y_va = get_X_y(val)
    X_te, y_te = get_X_y(test)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    models = available_models(cfg["training"]["models"])
    if len(models) < len(cfg["training"]["models"]):
        log.warning("Skipping unavailable models: %s", sorted(set(cfg["training"]["models"]) - set(models)))

    # ------------------------------------------------------------ Experiment 2: baseline vs engineered
    log.info("Experiment 2: baseline vs engineered features (default hyperparameters)")
    exp2 = []
    default_val = {}
    for name in models:
        for fs in ["baseline", "engineered"]:
            pipe = build_pipeline(name, seed, feature_set=fs, pos_weight=pos_weight)
            cv_m, cv_s = cross_validate_pipeline(pipe, X_tr, y_tr, cfg)
            pipe.fit(X_tr, y_tr)
            p = pipe.predict_proba(X_va)[:, 1]
            m = compute_metrics(y_va, p, find_best_threshold(y_va, p))
            exp2.append({"model": MODEL_DISPLAY_NAMES[name], "feature_set": fs, "n_features": len(feature_list(fs)),
                         "cv_pr_auc_mean": cv_m, "cv_pr_auc_std": cv_s, "val_pr_auc": m["pr_auc"],
                         "val_roc_auc": m["roc_auc"], "val_f1_at_val_threshold": m["f1"], "val_recall": m["recall"],
                         "val_precision": m["precision"]})
            default_val[(name, fs)] = m
            log.info("  %-22s %-10s CV PR-AUC %.4f±%.4f | val PR-AUC %.4f", MODEL_DISPLAY_NAMES[name], fs, cv_m,
                     cv_s, m["pr_auc"])
    exp2 = pd.DataFrame(exp2)
    exp2.to_csv(reps / "exp2_feature_sets.csv", index=False)
    fs_mean = exp2.groupby("feature_set")["val_pr_auc"].mean()
    feature_set = cfg["features"]["final_feature_set"]
    if feature_set == "auto":
        feature_set = fs_mean.idxmax()
    log.info("Mean val PR-AUC by feature set: %s -> using '%s'", fs_mean.round(4).to_dict(), feature_set)

    # ------------------------------------------------------------ Experiment 1 + 4: tuning
    log.info("Experiments 1 & 4: hyperparameter tuning (RandomizedSearchCV, %d iter, %d-fold, scoring=%s)",
             cfg["training"]["search_iterations"], cfg["training"]["cv_folds"], cfg["training"]["scoring"])
    tuned, tuning_info, val_rows, val_probas, thresholds = {}, {}, [], {}, {}
    for name in models:
        est, info, cv_res = tune_model(name, X_tr, y_tr, cfg, feature_set, logger=log)
        cv_res.drop(columns=[c for c in cv_res.columns if c.startswith("param_")], errors="ignore") \
            .to_csv(reps / "tuning" / f"{name}_cv_results.csv", index=False)
        tuned[name], tuning_info[name] = est, info
        p = est.predict_proba(X_va)[:, 1]
        thr = find_best_threshold(y_va, p, cfg["threshold"]["strategy"], cfg["threshold"]["fixed_value"])
        thresholds[name], val_probas[name] = thr, p
        m = compute_metrics(y_va, p, thr)
        d = default_val[(name, feature_set)]
        val_rows.append({"model": MODEL_DISPLAY_NAMES[name], "key": name, "cv_pr_auc": info["best_cv_score"],
                         "cv_pr_auc_std": info["best_cv_std"], **{f"val_{k}": m[k] for k in KEY_METRICS},
                         "val_threshold": thr, "default_val_pr_auc": d["pr_auc"], "default_val_f1": d["f1"],
                         "tuning_gain_pr_auc": m["pr_auc"] - d["pr_auc"], "tuning_gain_f1": m["f1"] - d["f1"],
                         "search_time_s": info["search_time_s"]})
        joblib.dump(est, P["models_dir"] / "all_models" / f"{name}.joblib", compress=3)
    val_df = pd.DataFrame(val_rows).sort_values("val_pr_auc", ascending=False)
    val_df.to_csv(reps / "validation_results.csv", index=False)
    save_json(tuning_info, reps / "best_hyperparameters.json")
    exp4 = val_df[["model", "default_val_pr_auc", "val_pr_auc", "tuning_gain_pr_auc", "default_val_f1", "val_f1",
                   "tuning_gain_f1"]]
    exp4.to_csv(reps / "exp4_tuning_impact.csv", index=False)

    sel_metric = "val_" + cfg["training"]["selection_metric"]
    best_name = val_df.sort_values(sel_metric, ascending=False).iloc[0]["key"]
    log.info("Model selection on VALIDATION %s -> %s", sel_metric, MODEL_DISPLAY_NAMES[best_name])

    # ------------------------------------------------------------ Experiment 3: imbalance strategies
    log.info("Experiment 3: class-imbalance strategies for %s (tuned hyperparameters)", MODEL_DISPLAY_NAMES[best_name])
    exp3 = []
    strat_pipes = {}
    for strat in supported_imbalance_strategies(best_name):
        if strat not in cfg["training"]["imbalance_strategies"]:
            continue
        pipe = set_imbalance_strategy(clone(tuned[best_name]), best_name, strat, seed, pos_weight)
        cv_m, cv_s = cross_validate_pipeline(pipe, X_tr, y_tr, cfg)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        thr = find_best_threshold(y_va, p)
        m = compute_metrics(y_va, p, thr)
        m05 = compute_metrics(y_va, p, 0.5)
        strat_pipes[strat] = pipe
        exp3.append({"strategy": strat, "cv_pr_auc_mean": cv_m, "cv_pr_auc_std": cv_s, "val_pr_auc": m["pr_auc"],
                     "val_f1_tuned_thr": m["f1"], "val_recall_tuned_thr": m["recall"],
                     "val_precision_tuned_thr": m["precision"], "val_threshold": thr,
                     "val_recall_at_0.5": m05["recall"], "val_precision_at_0.5": m05["precision"],
                     "val_brier": m["brier"]})
        log.info("  %-20s CV PR-AUC %.4f | val PR-AUC %.4f | val F1 %.4f | recall@0.5 %.3f", strat, cv_m,
                 m["pr_auc"], m["f1"], m05["recall"])
    exp3 = pd.DataFrame(exp3)
    exp3.to_csv(reps / "exp3_imbalance.csv", index=False)
    tuned_val_pr = float(val_df.set_index("key").loc[best_name, "val_pr_auc"])
    best_strat_row = exp3.sort_values("val_pr_auc", ascending=False).iloc[0]
    final_pipe = tuned[best_name]
    imbalance_choice = "as tuned (search included class weighting)"
    if best_strat_row["val_pr_auc"] > tuned_val_pr + 0.005:
        final_pipe = strat_pipes[best_strat_row["strategy"]]
        imbalance_choice = best_strat_row["strategy"]
        log.info("Adopting imbalance strategy '%s' (val PR-AUC %.4f > tuned %.4f)", imbalance_choice,
                 best_strat_row["val_pr_auc"], tuned_val_pr)
    else:
        log.info("Keeping tuned configuration (no strategy beat it by > 0.005 val PR-AUC)")

    # ------------------------------------------------------------ Experiment 5: calibration
    log.info("Experiment 5: probability calibration (fit with %d-fold CV on train, judged on validation)",
             cfg["training"]["cv_folds"])
    exp5, cal_models, cal_val_curves = [], {}, {}
    for method in cfg["calibration"]["methods"]:
        if method == "none":
            mdl = final_pipe
        else:
            mdl = CalibratedClassifierCV(clone(final_pipe), method=method, cv=cv_splitter(cfg)).fit(X_tr, y_tr)
        p = mdl.predict_proba(X_va)[:, 1]
        thr = find_best_threshold(y_va, p)
        m = compute_metrics(y_va, p, thr)
        cal_models[method] = mdl
        cal_val_curves[f"{method}"] = (y_va.to_numpy(), p, thr)
        exp5.append({"calibration": method, "val_brier": m["brier"], "val_ece": m["ece"], "val_log_loss": m["log_loss"],
                     "val_pr_auc": m["pr_auc"], "val_roc_auc": m["roc_auc"], "val_f1": m["f1"], "val_threshold": thr})
        log.info("  %-9s Brier %.5f | ECE %.5f | logloss %.4f | PR-AUC %.4f", method, m["brier"], m["ece"],
                 m["log_loss"], m["pr_auc"])
    exp5 = pd.DataFrame(exp5)
    exp5.to_csv(reps / "exp5_calibration.csv", index=False)
    cal_choice = exp5.sort_values("val_brier").iloc[0]["calibration"]
    deployed = cal_models[cal_choice]
    plot_calibration({f"{MODEL_DISPLAY_NAMES[best_name]} — {k}": v for k, v in cal_val_curves.items()},
                     figs / "calibration_methods_validation.png",
                     title="Calibration methods (validation set)")
    log.info("Calibration chosen by lowest validation Brier score: %s", cal_choice)

    p_val_dep = deployed.predict_proba(X_va)[:, 1]
    thr_dep = find_best_threshold(y_va, p_val_dep, cfg["threshold"]["strategy"], cfg["threshold"]["fixed_value"])
    log.info("Deployed decision threshold (validation, strategy=%s): %.4f", cfg["threshold"]["strategy"], thr_dep)

    # ------------------------------------------------------------ Final evaluation on TEST (once)
    log.info("Final evaluation on the held-out TEST set")
    test_rows, curves, test_pred_cols = [], {}, {"y_true": y_te.to_numpy()}
    for name in models:
        p = tuned[name].predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, p, thresholds[name])
        t = measure_inference_time(tuned[name], X_te)
        test_rows.append({"model": MODEL_DISPLAY_NAMES[name], "key": name, **{k: m[k] for k in KEY_METRICS},
                          "threshold": thresholds[name], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
                          "ms_per_sample_batch": t["ms_per_sample_batch"],
                          "single_row_latency_ms": t["single_row_latency_ms"],
                          "selected": name == best_name})
        curves[MODEL_DISPLAY_NAMES[name]] = (y_te.to_numpy(), p, thresholds[name])
        test_pred_cols[name] = p
    dep_label = f"{MODEL_DISPLAY_NAMES[best_name]} (deployed{', ' + cal_choice + ' calibration' if cal_choice != 'none' else ''})"
    p_te_dep = deployed.predict_proba(X_te)[:, 1]
    m_dep = compute_metrics(y_te, p_te_dep, thr_dep)
    t_dep = measure_inference_time(deployed, X_te)
    test_pred_cols["deployed"] = p_te_dep
    test_df = pd.DataFrame(test_rows)
    if cal_choice != "none":
        test_df = pd.concat([test_df, pd.DataFrame([{
            "model": dep_label, "key": "deployed", **{k: m_dep[k] for k in KEY_METRICS}, "threshold": thr_dep,
            "tn": m_dep["tn"], "fp": m_dep["fp"], "fn": m_dep["fn"], "tp": m_dep["tp"],
            "ms_per_sample_batch": t_dep["ms_per_sample_batch"], "single_row_latency_ms": t_dep["single_row_latency_ms"],
            "selected": True}])], ignore_index=True)
        test_df.loc[test_df["key"] == best_name, "selected"] = False
    test_df.to_csv(reps / "model_comparison_test.csv", index=False)
    save_json(test_df.to_dict(orient="records"), reps / "model_comparison_test.json")
    pd.DataFrame(test_pred_cols).to_csv(reps / "test_predictions.csv", index=False)
    pd.DataFrame({"y_true": y_va.to_numpy(), **val_probas, "deployed": p_val_dep}).to_csv(
        reps / "validation_predictions.csv", index=False)
    save_json({"deployed_model": dep_label, "threshold": thr_dep, "metrics": m_dep, "inference": t_dep,
               "classification_report": classification_report_dict(y_te, p_te_dep, thr_dep)},
              reps / "final_test_metrics.json")
    for _, r in test_df.iterrows():
        log.info("  TEST %-45s P %.3f R %.3f F1 %.3f PR-AUC %.3f ROC-AUC %.3f", r["model"], r["precision"],
                 r["recall"], r["f1"], r["pr_auc"], r["roc_auc"])

    # figures
    plot_confusion_matrices(curves, figs / "confusion_matrices_test.png")
    plot_confusion_matrices({dep_label: (y_te.to_numpy(), p_te_dep, thr_dep)}, figs / "confusion_matrix_deployed.png",
                            title="Deployed model — test set")
    plot_roc_curves(curves, figs / "roc_curves_test.png")
    plot_pr_curves(curves, figs / "pr_curves_test.png")
    plot_model_comparison(test_df[test_df["key"] != "deployed"], figs / "model_comparison_test.png")
    plot_calibration(curves, figs / "calibration_all_models_test.png",
                     title="Reliability diagram — uncalibrated tuned models (test)")
    plot_class_distribution(y_te, p_te_dep, thr_dep, figs / "actual_vs_predicted_distribution.png")
    plot_feature_importances({MODEL_DISPLAY_NAMES[n]: tuned[n] for n in models}, figs / "feature_importance_native.png")

    # ------------------------------------------------------------ Experiment 6: SHAP
    log.info("Experiment 6: SHAP interpretability analysis")
    explain_pipe = final_pipe  # uncalibrated pipeline fitted on train
    explainer = ShapExplainer(explain_pipe, X_tr, seed=seed, background_size=cfg["explainability"]["background_size"])
    n_shap = min(cfg["explainability"]["shap_sample_size"], len(X_te))
    X_shap = X_te.sample(n_shap, random_state=seed) if explainer.family != "neural_network" else X_te.sample(
        min(150, len(X_te)), random_state=seed)
    expl_raw = explainer.explain(X_shap)
    expl = group_onehot(expl_raw, raw_X=X_shap)
    gi = global_importance(expl)
    gi.to_csv(reps / "shap_global_importance.csv", index=False)
    mname = MODEL_DISPLAY_NAMES[best_name]
    save_summary_plot(expl, figs / "shap_summary.png", f"SHAP summary — {mname} ({explainer.output_space})")
    save_bar_plot(expl, figs / "shap_global_importance.png", f"Global SHAP importance — {mname}")
    top3 = [f for f in gi["feature"].head(3) if f != "type"][:3]
    save_dependence_plots(expl, top3, figs / "shap_dependence.png", "SHAP dependence plots (top features)")
    p_shap = deployed.predict_proba(X_shap)[:, 1]
    y_shap = y_te.loc[X_shap.index].to_numpy()
    pos_idx = np.where((y_shap == 1) & (p_shap >= thr_dep))[0]
    neg_idx = np.where((y_shap == 0) & (p_shap < thr_dep))[0]
    waterfalls = {}
    if len(pos_idx):
        i = pos_idx[np.argmax(p_shap[pos_idx])]
        save_waterfall_plot(expl[i], figs / "shap_waterfall_true_positive.png",
                            f"Waterfall — correctly flagged failure (p = {p_shap[i]:.3f})")
        waterfalls["true_positive"] = X_shap.iloc[i].to_dict() | {"probability": float(p_shap[i])}
    if len(neg_idx):
        i = neg_idx[np.argmin(p_shap[neg_idx])]
        save_waterfall_plot(expl[i], figs / "shap_waterfall_true_negative.png",
                            f"Waterfall — healthy machine (p = {p_shap[i]:.4f})")
        waterfalls["true_negative"] = X_shap.iloc[i].to_dict() | {"probability": float(p_shap[i])}
    np.savez_compressed(reps / "shap_values_test_sample.npz", values=expl.values, base_values=expl.base_values,
                        data=expl.data, feature_names=np.array(expl.feature_names, dtype=str),
                        index=X_shap.index.to_numpy(), y_true=y_shap, proba=p_shap)
    save_json({"model": mname, "output_space": explainer.output_space, "n_explained": int(n_shap),
               "global_importance": gi.to_dict(orient="records"), "waterfall_examples": waterfalls},
              reps / "exp6_shap.json")
    log.info("Top SHAP features: %s", gi[["feature", "share_pct"]].head(5).round(2).to_dict(orient="records"))

    # ------------------------------------------------------------ Experiment 7: error analysis
    log.info("Experiment 7: error analysis on test (deployed model)")
    err = test.copy()
    err["probability"] = p_te_dep
    err["pred"] = (p_te_dep >= thr_dep).astype(int)
    err["outcome"] = np.select([(err.pred == 1) & (err[TARGET] == 1), (err.pred == 0) & (err[TARGET] == 0),
                                (err.pred == 1) & (err[TARGET] == 0)], ["TP", "TN", "FP"], "FN")
    from src.feature_engineering import add_engineered_features
    err = add_engineered_features(err)
    err[err.outcome.isin(["FN", "FP"])].sort_values(["outcome", "probability"]).to_csv(
        reps / "exp7_misclassified_test.csv", index=False)
    num_cols = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min",
                "temp_diff_k", "power_w", "wear_torque_min_nm"]
    profile = err.groupby("outcome")[num_cols + ["probability"]].mean().round(3)
    profile["count"] = err.groupby("outcome").size()
    profile.to_csv(reps / "exp7_outcome_profiles.csv")
    fails = err[err[TARGET] == 1]
    mode_recall = []
    for mcol in FAILURE_MODE_COLUMNS:
        sub = fails[fails[mcol] == 1]
        mode_recall.append({"mode": FAILURE_MODE_NAMES[mcol], "failures_in_test": len(sub),
                            "detected": int((sub.pred == 1).sum()),
                            "recall": float((sub.pred == 1).mean()) if len(sub) else np.nan})
    no_mode = fails[fails[FAILURE_MODE_COLUMNS].sum(axis=1) == 0]
    mode_recall.append({"mode": "No mode label", "failures_in_test": len(no_mode), "detected": int((no_mode.pred == 1).sum()),
                        "recall": float((no_mode.pred == 1).mean()) if len(no_mode) else np.nan})
    mode_recall = pd.DataFrame(mode_recall)
    mode_recall.to_csv(reps / "exp7_recall_by_failure_mode.csv", index=False)
    fn_modes = err[err.outcome == "FN"][FAILURE_MODE_COLUMNS].sum().to_dict()
    fp_types = err[err.outcome == "FP"]["type"].value_counts().to_dict()
    log.info("FN count %d, FP count %d; recall by mode: %s", (err.outcome == "FN").sum(), (err.outcome == "FP").sum(),
             mode_recall.set_index("mode")["recall"].round(3).to_dict())

    # ------------------------------------------------------------ Ablation study
    log.info("Ablation: drop one feature at a time (%s, tuned hyperparameters, uncalibrated)", mname)
    abl = []
    base_feats = feature_list(feature_set)
    for drop in [None] + base_feats:
        pipe = clone(final_pipe)
        pipe.set_params(features__drop=[drop] if drop else None)
        pipe.named_steps["preprocess"].transformers = [
            (n, t, [c for c in cols if c != drop]) for n, t, cols in pipe.named_steps["preprocess"].transformers]
        cv_m, cv_s = cross_validate_pipeline(pipe, X_tr, y_tr, cfg)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        abl.append({"dropped_feature": drop or "(none — full set)", "cv_pr_auc_mean": cv_m, "cv_pr_auc_std": cv_s,
                    "val_pr_auc": compute_metrics(y_va, p)["pr_auc"]})
    abl = pd.DataFrame(abl)
    full = abl.iloc[0]["val_pr_auc"]
    abl["delta_val_pr_auc"] = abl["val_pr_auc"] - full
    abl.to_csv(reps / "ablation_study.csv", index=False)

    # ------------------------------------------------------------ Optional: failure modes
    log.info("Optional module: failure-mode (multi-label) models")
    fm_models, fm_df = train_failure_mode_models(train, val, test, seed)
    fm_df.to_csv(reps / "failure_mode_results.csv", index=False)
    overlap_table(df).to_csv(reps / "failure_mode_overlap.csv")
    joblib.dump(fm_models, P["models_dir"] / "failure_mode_models.joblib", compress=3)

    # ------------------------------------------------------------ Save deployed artifact
    ranges = {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in INPUT_COLUMNS if c != "type"}
    metadata = {
        "model_key": best_name, "model_display_name": mname, "deployed_label": dep_label,
        "feature_set": feature_set, "features": feature_list(feature_set), "feature_docs": FEATURE_DOCS,
        "imbalance_handling": imbalance_choice, "calibration": cal_choice, "decision_threshold": thr_dep,
        "threshold_strategy": cfg["threshold"]["strategy"], "best_params": tuning_info[best_name]["best_params"],
        "selection": f"highest validation {cfg['training']['selection_metric']}",
        "test_metrics": {k: m_dep[k] for k in KEY_METRICS}, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "library_versions": library_versions(), "seed": seed, "quick_run": args.quick,
        "train_rows": len(X_tr), "dataset": "AI4I 2020 Predictive Maintenance Dataset (UCI id 601, synthetic)",
    }
    artifact = {"model": deployed, "explain_pipeline": explain_pipe, "threshold": thr_dep,
                "background": X_tr.sample(cfg["explainability"]["background_size"], random_state=seed),
                "training_ranges": ranges, "metadata": metadata}
    joblib.dump(artifact, P["models_dir"] / "predictx_best_model.joblib", compress=3)
    save_json(metadata, P["models_dir"] / "model_card.json")

    # ------------------------------------------------------------ Experiment summary + markdown report
    experiments = {
        "exp1_model_comparison_validation": val_df.to_dict(orient="records"),
        "exp2_feature_sets": exp2.to_dict(orient="records"), "exp2_chosen_feature_set": feature_set,
        "exp3_imbalance": exp3.to_dict(orient="records"), "exp3_choice": imbalance_choice,
        "exp4_tuning_impact": exp4.to_dict(orient="records"),
        "exp5_calibration": exp5.to_dict(orient="records"), "exp5_choice": cal_choice,
        "exp7_recall_by_mode": mode_recall.to_dict(orient="records"),
        "exp7_fn_failure_modes": fn_modes, "exp7_fp_product_types": fp_types,
        "ablation": abl.to_dict(orient="records"), "failure_modes": fm_df.to_dict(orient="records"),
        "runtime_minutes": (time.perf_counter() - t_start) / 60,
    }
    save_json(experiments, reps / "experiments_summary.json")
    write_markdown_report(reps / "results_summary.md", metadata, split_info, eda_summary, exp2, val_df, exp3, exp4,
                          exp5, test_df, gi, mode_recall, profile, abl, fm_df, imbalance_choice, cal_choice,
                          experiments["runtime_minutes"])
    log.info("All artifacts saved. Total runtime %.1f min", experiments["runtime_minutes"])
    return metadata


def write_markdown_report(path, meta, split_info, eda, exp2, val_df, exp3, exp4, exp5, test_df, gi, mode_recall,
                          profile, abl, fm_df, imb, cal, runtime):
    T = lambda d, cols=None: md_table(d[cols] if cols else d)  # noqa: E731
    s = ["# PredictX AI — Experimental Results\n",
         f"_Generated automatically by `train.py` on {meta['trained_at']} (runtime {runtime:.1f} min, seed {meta['seed']}"
         f"{', QUICK RUN' if meta['quick_run'] else ''}). All numbers below come from this run._\n",
         "## Data split\n", md_table(pd.DataFrame(split_info).T.reset_index().rename(columns={"index": "split"})), "",
         "## EDA observations\n", *[f"- {o}" for o in eda["observations"]], "",
         "## Experiment 2 — baseline vs engineered features (default hyperparameters)\n",
         T(exp2, ["model", "feature_set", "cv_pr_auc_mean", "cv_pr_auc_std", "val_pr_auc", "val_f1_at_val_threshold"]),
         f"\nChosen feature set: **{meta['feature_set']}**\n",
         "## Experiment 1 — tuned model comparison (validation set, used for selection)\n",
         T(val_df, ["model", "cv_pr_auc", "val_pr_auc", "val_roc_auc", "val_precision", "val_recall", "val_f1",
                    "val_threshold"]),
         f"\nSelected model: **{meta['model_display_name']}** ({meta['selection']}).\n",
         "## Experiment 4 — impact of hyperparameter tuning (validation)\n", T(exp4), "",
         f"## Experiment 3 — class-imbalance strategies ({meta['model_display_name']})\n",
         T(exp3, ["strategy", "cv_pr_auc_mean", "val_pr_auc", "val_f1_tuned_thr", "val_recall_at_0.5",
                  "val_precision_at_0.5"]), f"\nDecision: **{imb}**\n",
         "## Experiment 5 — probability calibration (validation)\n", T(exp5), f"\nChosen: **{cal}** (lowest Brier)\n",
         "## Final held-out TEST results (thresholds tuned on validation)\n",
         T(test_df, ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate",
                     "false_negative_rate", "threshold", "single_row_latency_ms"]), "",
         "## Experiment 6 — SHAP global importance (deployed model, test sample)\n",
         T(gi, ["label", "mean_abs_shap", "share_pct"]), "",
         "## Experiment 7 — error analysis (test)\n", "Recall by failure mode:\n", T(mode_recall), "",
         "Mean feature values by outcome:\n", md_table(profile.reset_index()), "",
         "## Ablation (drop one feature)\n", T(abl), "",
         "## Optional module — failure-mode classifiers (one-vs-rest, test)\n",
         T(fm_df, ["mode", "name", "test_positives", "test_pr_auc", "test_no_skill_pr_auc", "test_roc_auc",
                   "test_recall", "test_precision"]), "",
         "## Caveats\n",
         "- AI4I 2020 is a **synthetic** benchmark whose labels follow documented rules; real plants are noisier.",
         "- SHAP shows model attributions, not proven physical causes.",
         "- Test-set sizes are small for failures (~68 cases), so metrics carry sampling uncertainty."]
    path.write_text("\n".join(s), encoding="utf-8")


if __name__ == "__main__":
    main()
