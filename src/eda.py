"""Exploratory data analysis: statistics, figures and data-driven observations."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import FAILURE_MODE_COLUMNS, FAILURE_MODE_NAMES, TARGET, validate_dataset
from .feature_engineering import BASE_NUMERIC, add_engineered_features

NAVY, CYAN, BLUE, RED, GREY = "#0B1F3A", "#00B4D8", "#1F6FEB", "#E5484D", "#8A94A6"
LABELS = {
    "air_temp_k": "Air temperature [K]",
    "process_temp_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
    "temp_diff_k": "Temp. difference [K]",
    "power_w": "Power [W]",
    "wear_torque_min_nm": "Wear x Torque [min·Nm]",
    "temp_diff_x_torque": "Temp. diff x Torque [K·Nm]",
}


def _style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 12, "figure.dpi": 110,
                         "savefig.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, out_dir: Path, name: str) -> str:
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def outlier_summary(df: pd.DataFrame, cols: list[str]) -> dict:
    """IQR (1.5×) outlier counts per numeric column."""
    res = {}
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[c] < lo) | (df[c] > hi)
        res[c] = {"lower_bound": float(lo), "upper_bound": float(hi), "n_outliers": int(mask.sum()),
                  "pct_outliers": float(100 * mask.mean()),
                  "failure_rate_in_outliers": float(df.loc[mask, TARGET].mean()) if mask.any() else None}
    return res


def compute_eda_summary(df: pd.DataFrame) -> dict:
    dfe = add_engineered_features(df)
    num = BASE_NUMERIC + ["temp_diff_k", "power_w", "wear_torque_min_nm"]
    validation = validate_dataset(df)
    by_type = df.groupby("type")[TARGET].agg(["mean", "sum", "count"]).reindex(["L", "M", "H"])
    corr = dfe[num + [TARGET]].corr()
    comp = dfe.groupby(TARGET)[num].mean().T.rename(columns={0: "no_failure_mean", 1: "failure_mean"})
    comp["ratio_failure_to_normal"] = comp["failure_mean"] / comp["no_failure_mean"]
    mode_counts = {FAILURE_MODE_NAMES[m]: int(df[m].sum()) for m in FAILURE_MODE_COLUMNS}
    co = df[FAILURE_MODE_COLUMNS].T.dot(df[FAILURE_MODE_COLUMNS])

    summary = {
        "shape": list(df.shape),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "validation": validation,
        "class_distribution": {"no_failure": int((df[TARGET] == 0).sum()), "failure": int(df[TARGET].sum()),
                               "failure_rate_pct": float(100 * df[TARGET].mean()),
                               "imbalance_ratio": float((df[TARGET] == 0).sum() / max(df[TARGET].sum(), 1))},
        "descriptive_statistics": dfe[num].describe().T.round(4).to_dict(orient="index"),
        "failure_rate_by_type": {t: {"failure_rate_pct": float(100 * r["mean"]), "failures": int(r["sum"]),
                                     "count": int(r["count"])} for t, r in by_type.iterrows()},
        "correlation_with_target": corr[TARGET].drop(TARGET).round(4).to_dict(),
        "air_vs_process_temp_corr": float(corr.loc["air_temp_k", "process_temp_k"]),
        "rpm_vs_torque_corr": float(corr.loc["rotational_speed_rpm", "torque_nm"]),
        "failure_vs_normal_means": comp.round(4).to_dict(orient="index"),
        "failure_mode_counts": mode_counts,
        "failure_mode_cooccurrence": {FAILURE_MODE_NAMES[a]: {FAILURE_MODE_NAMES[b]: int(co.loc[a, b])
                                      for b in FAILURE_MODE_COLUMNS} for a in FAILURE_MODE_COLUMNS},
        "outliers_iqr": outlier_summary(dfe, BASE_NUMERIC),
    }
    no_fail_mode = df[(df[TARGET] == 0) & (df[FAILURE_MODE_COLUMNS].sum(axis=1) > 0)][FAILURE_MODE_COLUMNS].sum()
    summary["mode_without_failure_breakdown"] = ", ".join(
        f"{m.upper()}={int(n)}" for m, n in no_fail_mode.items() if n > 0) or "none"
    summary["observations"] = generate_observations(summary)
    return summary


def generate_observations(s: dict) -> list[str]:
    """Plain-language observations built from the computed numbers (nothing hardcoded)."""
    cd, v = s["class_distribution"], s["validation"]
    fr = s["failure_rate_by_type"]
    comp = s["failure_vs_normal_means"]
    corr = s["correlation_with_target"]
    top_corr = sorted(corr.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    out = s["outliers_iqr"]
    obs = [
        f"The dataset has {s['shape'][0]:,} rows and {s['shape'][1]} columns with {v['total_missing']} missing values "
        f"and {v['duplicate_rows_all_columns']} fully duplicated rows.",
        f"Failures are rare: {cd['failure']} of {s['shape'][0]:,} records ({cd['failure_rate_pct']:.2f}%), an imbalance "
        f"of about 1:{cd['imbalance_ratio']:.0f}. Accuracy alone is therefore misleading (always predicting 'No Failure' "
        f"would already score {100 - cd['failure_rate_pct']:.1f}%).",
        "Failure rate by product type: " + ", ".join(f"{t} = {r['failure_rate_pct']:.2f}% (n={r['count']:,})"
                                                     for t, r in fr.items()) + ".",
        f"Failed machines have higher mean torque ({comp['torque_nm']['failure_mean']:.1f} vs "
        f"{comp['torque_nm']['no_failure_mean']:.1f} Nm) and higher mean tool wear "
        f"({comp['tool_wear_min']['failure_mean']:.0f} vs {comp['tool_wear_min']['no_failure_mean']:.0f} min).",
        f"Mean temperature difference is lower for failures ({comp['temp_diff_k']['failure_mean']:.2f} vs "
        f"{comp['temp_diff_k']['no_failure_mean']:.2f} K), consistent with heat-dissipation problems.",
        f"Rotational speed and torque are strongly negatively correlated (r = {s['rpm_vs_torque_corr']:.2f}); air and "
        f"process temperature are strongly positively correlated (r = {s['air_vs_process_temp_corr']:.2f}).",
        "Strongest linear correlations with failure: " + ", ".join(f"{LABELS.get(k, k)} ({val:+.3f})"
                                                                 for k, val in top_corr) +
        ". Linear correlations are weak overall, which suggests non-linear (threshold-like) failure behaviour.",
        "IQR outliers: " + ", ".join(
            f"{LABELS[c]} {o['n_outliers']} rows (failure rate {100 * o['failure_rate_in_outliers']:.1f}%)"
            for c, o in out.items() if o["n_outliers"] > 0) +
        f" versus {cd['failure_rate_pct']:.2f}% overall. These are plausible extreme operating points that carry "
        f"real failure signal, so they are kept rather than removed.",
        f"Failure-mode labels overlap: {v['rows_with_multiple_modes']} rows carry more than one mode, "
        f"{v['failures_without_mode_label']} failures have no mode label, and {v['mode_label_without_failure']} rows have "
        f"a mode flag but Machine failure = 0 ({s.get('mode_without_failure_breakdown', '')}).",
    ]
    return obs


def generate_eda_figures(df: pd.DataFrame, out_dir: str | Path) -> list[str]:
    _style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dfe = add_engineered_features(df)
    dfe["Status"] = dfe[TARGET].map({0: "No Failure", 1: "Failure"})
    pal = {"No Failure": BLUE, "Failure": RED}
    paths = []

    # 1. Failure distribution
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts = dfe["Status"].value_counts().reindex(["No Failure", "Failure"])
    ax[0].bar(counts.index, counts.values, color=[BLUE, RED])
    for i, val in enumerate(counts.values):
        ax[0].text(i, val, f"{val:,}\n({100 * val / len(dfe):.2f}%)", ha="center", va="bottom")
    ax[0].set_title("Machine failure distribution"); ax[0].set_ylabel("Records")
    ax[0].set_ylim(0, counts.max() * 1.18)
    modes = pd.Series({FAILURE_MODE_NAMES[m]: df[m].sum() for m in FAILURE_MODE_COLUMNS}).sort_values()
    ax[1].barh(modes.index, modes.values, color=CYAN)
    ax[1].set_title("Failure-mode label counts (modes can overlap)"); ax[1].set_xlabel("Records")
    paths.append(_save(fig, out_dir, "eda_failure_distribution.png"))

    # 2. Temperature distributions
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col in zip(ax, ["air_temp_k", "process_temp_k", "temp_diff_k"]):
        sns.histplot(data=dfe, x=col, hue="Status", palette=pal, stat="density", common_norm=False,
                     element="step", bins=40, ax=a)
        a.set_title(LABELS[col]); a.set_xlabel(LABELS[col])
    paths.append(_save(fig, out_dir, "eda_temperature_distributions.png"))

    # 3. Torque vs rotational speed
    fig, ax = plt.subplots(figsize=(8, 6))
    normal = dfe[dfe[TARGET] == 0]; fail = dfe[dfe[TARGET] == 1]
    ax.scatter(normal["rotational_speed_rpm"], normal["torque_nm"], s=6, alpha=0.25, c=BLUE, label="No Failure")
    ax.scatter(fail["rotational_speed_rpm"], fail["torque_nm"], s=18, alpha=0.85, c=RED, label="Failure")
    rpm = np.linspace(dfe["rotational_speed_rpm"].min(), dfe["rotational_speed_rpm"].max(), 200)
    for pw, ls in [(3500, "--"), (9000, ":")]:
        ax.plot(rpm, pw / (rpm * 2 * np.pi / 60), color=NAVY, ls=ls, lw=1.2, label=f"Power = {pw} W")
    ax.set_ylim(0, dfe["torque_nm"].max() * 1.05)
    ax.set_xlabel(LABELS["rotational_speed_rpm"]); ax.set_ylabel(LABELS["torque_nm"])
    ax.set_title("Torque vs rotational speed"); ax.legend(loc="upper right")
    paths.append(_save(fig, out_dir, "eda_torque_vs_rpm.png"))

    # 4. Tool wear distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=dfe, x="tool_wear_min", hue="Status", palette=pal, bins=50, stat="density",
                 common_norm=False, element="step", ax=ax[0])
    ax[0].set_title("Tool wear distribution")
    bins = pd.cut(dfe["tool_wear_min"], bins=range(0, 280, 20), right=False)
    rate = dfe.groupby(bins, observed=False)[TARGET].mean() * 100
    ax[1].bar([str(b.left) for b in rate.index], rate.values, color=CYAN)
    ax[1].set_title("Failure rate by tool-wear bin (20 min)"); ax[1].set_ylabel("Failure rate [%]")
    ax[1].set_xlabel("Tool wear bin start [min]"); ax[1].tick_params(axis="x", rotation=45)
    paths.append(_save(fig, out_dir, "eda_tool_wear_distribution.png"))

    # 5. Correlation heatmap
    cols = BASE_NUMERIC + ["temp_diff_k", "power_w", "wear_torque_min_nm", TARGET] + FAILURE_MODE_COLUMNS
    corr = dfe[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": .8},
                annot_kws={"size": 7}, ax=ax,
                xticklabels=[LABELS.get(c, c) for c in cols], yticklabels=[LABELS.get(c, c) for c in cols])
    ax.set_title("Correlation heatmap (Pearson)")
    paths.append(_save(fig, out_dir, "eda_correlation_heatmap.png"))

    # 6. Failure rate by product type
    fig, ax = plt.subplots(figsize=(7, 4))
    g = df.groupby("type")[TARGET].agg(["mean", "count"]).reindex(["L", "M", "H"])
    bars = ax.bar(g.index, g["mean"] * 100, color=[CYAN, BLUE, NAVY])
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{100 * r['mean']:.2f}%\n(n={int(r['count']):,})",
                ha="center", va="bottom")
    ax.set_ylim(0, g["mean"].max() * 100 * 1.3)
    ax.set_title("Failure rate by product type (L / M / H quality)"); ax.set_ylabel("Failure rate [%]")
    paths.append(_save(fig, out_dir, "eda_failure_rate_by_type.png"))

    # 7. Feature comparison by failure status
    comp_cols = BASE_NUMERIC + ["temp_diff_k", "power_w", "wear_torque_min_nm"]
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    for a, col in zip(ax.flat, comp_cols):
        sns.boxplot(data=dfe, x="Status", y=col, hue="Status", palette=pal, ax=a, legend=False,
                    order=["No Failure", "Failure"], fliersize=2)
        a.set_title(LABELS[col]); a.set_xlabel(""); a.set_ylabel("")
    fig.suptitle("Feature comparison by failure status (boxes show IQR; dots are IQR outliers)", fontweight="bold")
    fig.tight_layout()
    paths.append(_save(fig, out_dir, "eda_feature_comparison_by_failure.png"))

    # 8. Feature relationships tied to documented failure mechanisms
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for status, color, s in [("No Failure", BLUE, 5), ("Failure", RED, 16)]:
        d = dfe[dfe["Status"] == status]
        ax[0].scatter(d["rotational_speed_rpm"], d["temp_diff_k"], s=s, alpha=0.3 if status == "No Failure" else 0.85,
                      c=color, label=status)
        ax[1].scatter(d["tool_wear_min"], d["torque_nm"], s=s, alpha=0.3 if status == "No Failure" else 0.85,
                      c=color, label=status)
    ax[0].set_xlabel(LABELS["rotational_speed_rpm"]); ax[0].set_ylabel(LABELS["temp_diff_k"])
    ax[0].set_title("Temperature difference vs speed"); ax[0].legend()
    ax[1].set_xlabel(LABELS["tool_wear_min"]); ax[1].set_ylabel(LABELS["torque_nm"])
    ax[1].set_title("Torque vs tool wear"); ax[1].legend()
    paths.append(_save(fig, out_dir, "eda_feature_relationships.png"))
    return paths
