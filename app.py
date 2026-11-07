"""PredictX AI — Intelligent Predictive Maintenance (Streamlit dashboard).

    streamlit run app.py

Every number and chart on this dashboard is read from the real dataset or from the
artifacts written by train.py. Nothing is hardcoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.data_loader import FAILURE_MODE_COLUMNS, FAILURE_MODE_NAMES, INPUT_COLUMNS, REVERSE_COLUMN_MAP, TARGET, load_dataset
from src.explainability import PRETTY
from src.feature_engineering import add_engineered_features
from src.inference import PredictXPredictor
from src.utils import InputValidationError, ModelNotFoundError, load_config, load_json, resolve_path

st.set_page_config(page_title="PredictX AI — Intelligent Predictive Maintenance", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="expanded")

CFG = load_config()
REPORTS = resolve_path(CFG["paths"]["reports_dir"])
FIGS = resolve_path(CFG["paths"]["figures_dir"])
CYAN, BLUE, NAVY, RED, AMBER, GREEN, MUTED = "#22D3EE", "#3B82F6", "#0E2244", "#F43F5E", "#F59E0B", "#10B981", "#8DA2C0"
RISK_COLORS = {"Low": GREEN, "Medium": AMBER, "High": RED}
SEQ = [CYAN, BLUE, "#A78BFA", AMBER, GREEN, RED, "#F472B6"]
px.defaults.color_discrete_sequence = SEQ

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #0F2A55 0%, #07142B 55%) fixed; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#081A36 0%,#061226 100%); border-right:1px solid #16305A; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.hero { padding: 22px 28px; border-radius: 18px; border:1px solid #1C3A6B;
        background: linear-gradient(120deg, rgba(34,211,238,.12), rgba(59,130,246,.08) 45%, rgba(14,34,68,.6));
        margin-bottom: 18px; }
.hero h1 { margin:0; font-size: 2.0rem; font-weight:800;
           background: linear-gradient(90deg,#E6EEF8,#22D3EE); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { margin:.35rem 0 0 0; color:#A9BCD8; font-size:.98rem; }
.kpi { background: linear-gradient(180deg,#0F2547,#0B1C38); border:1px solid #1A3766; border-radius:14px;
       padding:14px 16px; height:100%; position:relative; overflow:hidden; }
.kpi:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(#22D3EE,#3B82F6); }
.kpi .label { color:#8DA2C0; font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
.kpi .value { color:#F1F6FD; font-size:1.55rem; font-weight:700; margin-top:2px; font-family:'JetBrains Mono',monospace; }
.kpi .sub { color:#6F86A8; font-size:.78rem; margin-top:2px; }
.pill { display:inline-block; padding:3px 12px; border-radius:999px; font-weight:700; font-size:.85rem; }
.panel { background:#0B1C38; border:1px solid #1A3766; border-radius:14px; padding:16px 18px; }
.note { color:#8DA2C0; font-size:.82rem; }
.rec-high { border-left:4px solid #F43F5E; } .rec-medium { border-left:4px solid #F59E0B; } .rec-low { border-left:4px solid #10B981; }
div[data-testid="stMetric"] { background:#0B1C38; border:1px solid #1A3766; border-radius:12px; padding:10px 14px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background:#0B1C38; border-radius:10px 10px 0 0; padding:8px 14px; }
</style>
""", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def kpi_row(items: list[tuple]):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(kpi(*it), unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,28,56,.55)",
                      font=dict(family="Inter, sans-serif", color="#D5E1F2", size=12), height=height,
                      margin=dict(l=10, r=10, t=48, b=10), colorway=SEQ,
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#1A3361", zerolinecolor="#1A3361")
    fig.update_yaxes(gridcolor="#1A3361", zerolinecolor="#1A3361")
    return fig


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return add_engineered_features(load_dataset(CFG, allow_download=True))


@st.cache_resource(show_spinner="Loading trained model…")
def get_predictor():
    return PredictXPredictor(cfg=CFG)


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame | None:
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_json(name: str):
    p = REPORTS / name
    return load_json(p) if p.exists() else None


def need_artifacts() -> bool:
    try:
        get_predictor()
        return True
    except ModelNotFoundError as e:
        st.error(f"{e}\n\nTrain the models first:  `python train.py`  — the dashboard only shows real results.")
        return False


def fig_image(name: str, caption: str = ""):
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not found — run `python train.py`.")


# ----------------------------------------------------------------------------- sidebar
PAGES = ["Overview", "Machine Failure Prediction", "Batch Prediction", "Model Performance", "Explainable AI",
         "Dataset Analytics"]
ICONS = ["🏠", "🔮", "📦", "📊", "🧠", "🔎"]
with st.sidebar:
    st.markdown('<div style="font-size:1.45rem;font-weight:800;color:#E6EEF8">⚙️ PredictX <span style="color:#22D3EE">AI</span></div>'
                '<div class="note" style="margin-bottom:14px">Intelligent Predictive Maintenance</div>',
                unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, format_func=lambda p: f"{ICONS[PAGES.index(p)]}  {p}",
                    label_visibility="collapsed")
    st.markdown("---")
    try:
        _pred = get_predictor()
        md = _pred.metadata
        st.markdown(
            f'<div class="panel"><div class="note">DEPLOYED MODEL</div>'
            f'<div style="font-weight:700;color:#22D3EE;margin:2px 0 6px 0">{md["model_display_name"]}</div>'
            f'<div class="note">Features: {md["feature_set"]} ({len(md["features"])})<br>'
            f'Calibration: {md["calibration"]}<br>Decision threshold: {md["decision_threshold"]:.3f}<br>'
            f'Trained: {md["trained_at"].replace("T", " ")}</div></div>', unsafe_allow_html=True)
    except ModelNotFoundError:
        st.warning("No trained model found. Run `python train.py`.")
    st.markdown('<div class="note" style="margin-top:14px">Dataset: AI4I 2020 (UCI #601) — a <b>synthetic</b> '
                'benchmark, not live plant data. Predictions are advisory.</div>', unsafe_allow_html=True)


# ============================================================================= PAGE 1
def page_overview():
    hero("PredictX AI — Intelligent Predictive Maintenance",
         "An explainable machine-learning framework that estimates industrial equipment failure risk from operating "
         "conditions, explains every prediction with SHAP and turns it into advisory maintenance actions.")
    df = get_dataset()
    eda = read_json("eda_summary.json")
    test = read_csv("model_comparison_test.csv")
    n_fail = int(df[TARGET].sum())
    items = [("Records", f"{len(df):,}", "AI4I 2020 machines"),
             ("Failures", f"{n_fail}", f"{100 * n_fail / len(df):.2f}% failure rate"),
             ("Input features", "6", "type, 2 temps, speed, torque, wear")]
    if test is not None:
        sel = test[test["selected"]].iloc[0]
        items += [("Test PR-AUC", f"{sel['pr_auc']:.3f}", sel["model"]),
                  ("Test recall", f"{sel['recall']:.3f}", f"precision {sel['precision']:.3f}"),
                  ("Test F1", f"{sel['f1']:.3f}", f"threshold {sel['threshold']:.3f}")]
    kpi_row(items)
    st.write("")
    c1, c2 = st.columns([1, 1.25])
    with c1:
        counts = df[TARGET].map({0: "No Failure", 1: "Failure"}).value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.62,
                               marker=dict(colors=[BLUE, RED]), textinfo="percent", textposition="outside"))
        fig.update_layout(title="Machine failure distribution", showlegend=True,
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with c2:
        modes = pd.DataFrame({"mode": [FAILURE_MODE_NAMES[m] for m in FAILURE_MODE_COLUMNS],
                              "count": [int(df[m].sum()) for m in FAILURE_MODE_COLUMNS]}).sort_values("count")
        fig = px.bar(modes, x="count", y="mode", orientation="h", title="Failure-mode labels (can overlap)",
                     color_discrete_sequence=[CYAN], text="count")
        st.plotly_chart(style_fig(fig, 340), width="stretch")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("#### Key insights from the data")
        if eda:
            for o in eda["observations"]:
                st.markdown(f"- {o}")
    with c2:
        st.markdown("#### How the system works")
        st.markdown("""
1. **Inputs** — product type and five operating measurements.
2. **Feature engineering** — temperature gap, mechanical power, wear × torque, temp-gap × torque.
3. **Model** — the best of six tuned algorithms, selected on validation PR-AUC.
4. **Threshold & risk bands** — tuned on validation, configurable in `config.yaml`.
5. **Explanation** — SHAP shows which inputs pushed the risk up or down.
6. **Recommendation** — documented rules + model output → advisory action.
""")
        try:
            md = get_predictor().metadata
            info = pd.DataFrame({"Item": ["Algorithm", "Selection rule", "Feature set", "Imbalance handling",
                                          "Calibration", "Decision threshold", "Training rows"],
                                 "Value": [md["model_display_name"], md["selection"], md["feature_set"],
                                           md["imbalance_handling"], md["calibration"],
                                           f"{md['decision_threshold']:.4f}", f"{md['train_rows']:,}"]})
            st.dataframe(info, hide_index=True, width="stretch")
        except ModelNotFoundError:
            pass


# ============================================================================= PAGE 2
def risk_gauge(p: float, low: float, high: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p, number={"suffix": "%", "valueformat": ".2f", "font": {"size": 40}},
        title={"text": "Estimated failure probability"},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED}, "bar": {"color": CYAN, "thickness": .28},
               "bgcolor": "rgba(0,0,0,0)",
               "steps": [{"range": [0, 100 * low], "color": "rgba(16,185,129,.35)"},
                         {"range": [100 * low, 100 * high], "color": "rgba(245,158,11,.35)"},
                         {"range": [100 * high, 100], "color": "rgba(244,63,94,.38)"}],
               "threshold": {"line": {"color": "#FFFFFF", "width": 3}, "thickness": .8, "value": 100 * high}}))
    return style_fig(fig, 300)


def shap_waterfall_fig(shap_payload: dict, top_n: int = 10) -> go.Figure:
    vals = np.array(shap_payload["values"]); names = shap_payload["feature_names"]
    order = np.argsort(-np.abs(vals))
    keep, rest = order[:top_n], order[top_n:]
    labels = [PRETTY.get(names[i], names[i]) for i in keep]
    contrib = list(vals[keep])
    if len(rest):
        labels.append(f"{len(rest)} other features"); contrib.append(float(vals[rest].sum()))
    base = shap_payload["base_value"]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=["Base value (avg model output)"] + labels[::-1] + ["Model output f(x)"],
        x=[base] + contrib[::-1] + [0], measure=["absolute"] + ["relative"] * len(contrib) + ["total"],
        increasing={"marker": {"color": RED}}, decreasing={"marker": {"color": BLUE}},
        totals={"marker": {"color": CYAN}}, connector={"line": {"color": "#36507A"}},
        text=[f"{base:+.2f}"] + [f"{c:+.3f}" for c in contrib[::-1]] + [f"{base + sum(contrib):+.2f}"],
        textposition="outside"))
    fig.update_layout(title=f"SHAP waterfall ({shap_payload['output_space']} space)", showlegend=False)
    return style_fig(fig, 460)


def page_prediction():
    hero("Machine Failure Prediction",
         "Enter the operating conditions of one machine. The trained model returns the failure probability, a risk "
         "category, the inputs that drove the score and an advisory maintenance action.")
    if not need_artifacts():
        return
    pred = get_predictor()
    ranges = pred.training_ranges
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        mtype = c1.selectbox("Machine product type", ["L", "M", "H"], index=1,
                             help="Product quality variant: L (low, 50% of data), M (medium, 30%), H (high, 20%)")
        air = c2.number_input("Air temperature [K]", value=300.0, step=0.1, format="%.1f",
                              help=f"Training range {ranges['air_temp_k'][0]:.1f}–{ranges['air_temp_k'][1]:.1f} K")
        proc = c3.number_input("Process temperature [K]", value=310.0, step=0.1, format="%.1f",
                               help=f"Training range {ranges['process_temp_k'][0]:.1f}–{ranges['process_temp_k'][1]:.1f} K")
        c4, c5, c6 = st.columns(3)
        rpm = c4.number_input("Rotational speed [rpm]", value=1500.0, step=10.0, format="%.0f",
                              help=f"Training range {ranges['rotational_speed_rpm'][0]:.0f}–{ranges['rotational_speed_rpm'][1]:.0f}")
        torque = c5.number_input("Torque [Nm]", value=40.0, step=0.5, format="%.1f",
                                 help=f"Training range {ranges['torque_nm'][0]:.1f}–{ranges['torque_nm'][1]:.1f} Nm")
        wear = c6.number_input("Tool wear [min]", value=120.0, step=1.0, format="%.0f",
                               help=f"Training range {ranges['tool_wear_min'][0]:.0f}–{ranges['tool_wear_min'][1]:.0f} min")
        submitted = st.form_submit_button("⚡  Predict failure risk", type="primary", width="stretch")

    if not submitted and "last_pred" not in st.session_state:
        st.markdown('<div class="note">Tip: try Type L, 298 K / 308.5 K, 1300 rpm, 62 Nm, 215 min to see a high-risk case.</div>',
                    unsafe_allow_html=True)
        return
    if submitted:
        try:
            st.session_state["last_pred"] = pred.predict({"type": mtype, "air_temp_k": air, "process_temp_k": proc,
                                                          "rotational_speed_rpm": rpm, "torque_nm": torque,
                                                          "tool_wear_min": wear})
        except InputValidationError as e:
            st.error(f"Invalid input: {e}")
            return
    res = st.session_state["last_pred"]
    for w in res["warnings"]:
        st.warning(w)
    cat = res["risk_category"]
    color = RISK_COLORS[cat]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Predicted class", res["predicted_class"], f"threshold {res['decision_threshold']:.3f}"),
                unsafe_allow_html=True)
    c2.markdown(kpi("Failure probability", f"{100 * res['failure_probability']:.2f}%", res["model_name"]),
                unsafe_allow_html=True)
    c3.markdown(kpi("Risk category", f'<span class="pill" style="background:{color}22;color:{color};'
                    f'border:1px solid {color}">{cat}</span>',
                    f"Low < {res['risk_thresholds']['low_max']:.2f} ≤ Medium < {res['risk_thresholds']['high_min']:.3f} ≤ High"),
                unsafe_allow_html=True)
    c4.markdown(kpi("Maintenance priority", res["recommendation"]["priority"], "rule-based, advisory"),
                unsafe_allow_html=True)
    st.write("")
    g1, g2 = st.columns([1, 1.35])
    with g1:
        st.plotly_chart(risk_gauge(res["failure_probability"], res["risk_thresholds"]["low_max"],
                                   res["risk_thresholds"]["high_min"]), width="stretch")
        rec = res["recommendation"]
        st.markdown(f'<div class="panel rec-{cat.lower()}"><b>{rec["headline"]}</b><ul>' +
                    "".join(f"<li>{a}</li>" for a in rec["actions"]) + "</ul>" +
                    ("".join(f'<div class="note">• [{c["severity"]}] {c["mechanism"]}: {c["detail"]}</div>'
                             for c in rec["rule_checks"])) +
                    f'<div class="note" style="margin-top:8px"><i>{rec["disclaimer"]}</i></div></div>',
                    unsafe_allow_html=True)
    with g2:
        top = pd.DataFrame(res["top_features"])
        top["shown"] = top.apply(lambda r: f"{r['label']} = {r['value'] if isinstance(r['value'], str) else round(r['value'], 2)}",
                                 axis=1)
        fig = go.Figure(go.Bar(x=top["shap_value"][::-1], y=top["shown"][::-1], orientation="h",
                               marker_color=[RED if v > 0 else BLUE for v in top["shap_value"][::-1]],
                               text=[f"{v:+.3f}" for v in top["shap_value"][::-1]], textposition="auto",
                               cliponaxis=False))
        fig.update_layout(title="Most influential inputs (red ↑ risk, blue ↓ risk)", xaxis_title="SHAP contribution")
        st.plotly_chart(style_fig(fig, 330), width="stretch")
        st.markdown('<div class="note">SHAP values show how the <b>model</b> used each input for this prediction. '
                    'They are statistical attributions, not proven physical causes of failure.</div>',
                    unsafe_allow_html=True)
    with st.expander("Full SHAP waterfall for this prediction", expanded=False):
        st.plotly_chart(shap_waterfall_fig(res["shap"]), width="stretch")
    if res.get("failure_modes"):
        with st.expander("Optional module — failure-mode probabilities (independent multi-label models)"):
            fm = pd.DataFrame({"mode": list(res["failure_modes"]), "probability": list(res["failure_modes"].values())})
            fig = px.bar(fm, x="probability", y="mode", orientation="h", text=fm["probability"].map(lambda v: f"{v:.3f}"),
                         color_discrete_sequence=[CYAN])
            fig.update_xaxes(range=[0, 1])
            st.plotly_chart(style_fig(fig, 280), width="stretch")
            st.caption("Each mode has its own classifier; modes can overlap, so these probabilities do not sum to 1. "
                       "Random failures (RNF) are by design not predictable from the inputs.")


# ============================================================================= PAGE 3
def page_batch():
    hero("Batch Prediction", "Upload a CSV of machines and score them all at once. Each row is validated; invalid rows "
                             "are reported instead of silently dropped.")
    if not need_artifacts():
        return
    pred = get_predictor()
    template = pd.DataFrame([{REVERSE_COLUMN_MAP[c]: v for c, v in zip(INPUT_COLUMNS, ["M", 300.0, 310.0, 1500, 40.0, 120])},
                             {REVERSE_COLUMN_MAP[c]: v for c, v in zip(INPUT_COLUMNS, ["L", 298.0, 308.5, 1300, 62.0, 215])}])
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Upload CSV", type=["csv"],
                              help="Required columns: Type, Air temperature [K], Process temperature [K], "
                                   "Rotational speed [rpm], Torque [Nm], Tool wear [min] (snake_case names also work).")
    with c2:
        st.download_button("⬇ Download CSV template", template.to_csv(index=False), "predictx_template.csv", "text/csv",
                           width="stretch")
        use_test = st.button("Use 500 rows of the held-out test set", width="stretch")
    df = None
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read CSV: {e}")
    elif use_test or st.session_state.get("batch_source") == "test":
        st.session_state["batch_source"] = "test"
        p = resolve_path(CFG["paths"]["processed_dir"]) / "test.csv"
        df = pd.read_csv(p).sample(500, random_state=CFG["seed"]).reset_index(drop=True)
        st.caption("Scoring a real sample of the held-out test split (labels shown for comparison only).")
    if df is None:
        return
    try:
        with st.spinner(f"Scoring {len(df):,} rows…"):
            out = pred.predict_batch(df, explain=len(df) <= 5000)
    except InputValidationError as e:
        st.error(str(e))
        return
    valid = out[out["validation_error"] == ""]
    n_bad = len(out) - len(valid)
    rc = valid["risk_category"].value_counts()
    kpi_row([("Rows scored", f"{len(valid):,}", f"{n_bad} invalid rows"),
             ("Predicted failures", f"{(valid['predicted_class'] == 'Failure').sum()}", ""),
             ("High risk", f"{rc.get('High', 0)}", ""), ("Medium risk", f"{rc.get('Medium', 0)}", ""),
             ("Low risk", f"{rc.get('Low', 0)}", "")])
    if n_bad:
        st.warning(f"{n_bad} rows failed validation — see the `validation_error` column.")
    c1, c2 = st.columns(2)
    with c1:
        d = valid["risk_category"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).reset_index()
        d.columns = ["risk", "count"]
        fig = px.bar(d, x="risk", y="count", color="risk", color_discrete_map=RISK_COLORS, text="count",
                     title="Risk distribution")
        st.plotly_chart(style_fig(fig, 330), width="stretch")
    with c2:
        fig = px.histogram(valid, x="failure_probability", nbins=40, title="Predicted probability distribution",
                           color_discrete_sequence=[CYAN], log_y=True)
        st.plotly_chart(style_fig(fig, 330), width="stretch")
    if "Machine failure" in out.columns or TARGET in out.columns:
        lab = "Machine failure" if "Machine failure" in out.columns else TARGET
        v = valid.copy(); v["actual"] = v[lab].map({0: "No Failure", 1: "Failure"})
        ct = pd.crosstab(v["actual"], v["predicted_class"])
        st.markdown("**Labels were present in the file — actual vs predicted:**")
        st.dataframe(ct, width="stretch")
    show = st.selectbox("Filter table", ["All", "High", "Medium", "Low", "Invalid rows"])
    view = out if show == "All" else (out[out["validation_error"] != ""] if show == "Invalid rows"
                                      else out[out["risk_category"] == show])
    st.dataframe(view.sort_values("failure_probability", ascending=False), width="stretch", height=380,
                 column_config={"failure_probability": st.column_config.ProgressColumn(
                     "failure_probability", format="%.4f", min_value=0.0, max_value=1.0)})
    st.download_button("⬇ Download prediction results (CSV)", out.to_csv(index=False), "predictx_batch_predictions.csv",
                        "text/csv", type="primary")


# ============================================================================= PAGE 4
def page_performance():
    hero("Model Performance", "Six tuned algorithms compared on the same held-out test split. Model selection and "
                              "thresholds used the validation split only.")
    test = read_csv("model_comparison_test.csv")
    preds = read_csv("test_predictions.csv")
    if test is None or preds is None:
        st.error("Run `python train.py` to generate evaluation results.")
        return
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate",
            "false_negative_rate", "threshold", "single_row_latency_ms", "selected"]
    st.markdown("#### Test-set comparison")
    st.dataframe(test[cols].style.format({c: "{:.4f}" for c in cols[1:-2]} | {"single_row_latency_ms": "{:.2f}"})
                 .highlight_max(subset=["precision", "recall", "f1", "roc_auc", "pr_auc"], color="#0E4D63"),
                 hide_index=True, width="stretch")
    st.caption("Accuracy is reported but not used for decisions: with 3.4% failures, a model that never predicts "
               "failure would already reach ~96.6% accuracy.")
    metrics = st.multiselect("Metrics", ["precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy"],
                             default=["precision", "recall", "f1", "pr_auc"])
    if metrics:
        long = test.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
        fig = px.bar(long, x="model", y="score", color="metric", barmode="group", text=long["score"].round(3),
                     title="Model comparison (test)")
        fig.update_yaxes(range=[0, 1.08])
        st.plotly_chart(style_fig(fig, 420), width="stretch")

    key_to_label = dict(zip(test["key"], test["model"]))
    c1, c2 = st.columns(2)
    roc, pr = go.Figure(), go.Figure()
    y = preds["y_true"].to_numpy()
    for k, lab in key_to_label.items():
        if k not in preds.columns:
            continue
        fpr, tpr, _ = roc_curve(y, preds[k]); prec, rec, _ = precision_recall_curve(y, preds[k])
        r = test[test["key"] == k].iloc[0]
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{lab} ({r['roc_auc']:.3f})"))
        pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{lab} ({r['pr_auc']:.3f})"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED), name="Chance"))
    pr.add_hline(y=y.mean(), line_dash="dash", line_color=MUTED, annotation_text=f"No-skill {y.mean():.3f}")
    roc.update_layout(title="ROC curves (test)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    pr.update_layout(title="Precision-recall curves (test)", xaxis_title="Recall", yaxis_title="Precision")
    c1.plotly_chart(style_fig(roc, 430), width="stretch")
    c2.plotly_chart(style_fig(pr, 430), width="stretch")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        choice = st.selectbox("Confusion matrix for", list(key_to_label.values()),
                              index=int(np.where(test["selected"])[0][0]))
        k = test[test["model"] == choice]["key"].iloc[0]
        thr = float(test[test["model"] == choice]["threshold"].iloc[0])
        cm = confusion_matrix(y, (preds[k] >= thr).astype(int), labels=[0, 1])
        fig = px.imshow(cm, text_auto=True, x=["No Failure", "Failure"], y=["No Failure", "Failure"],
                        color_continuous_scale=[[0, "#0B1C38"], [1, CYAN]], labels=dict(x="Predicted", y="Actual"),
                        title=f"{choice} — threshold {thr:.3f}")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 400), width="stretch")
    with c2:
        fig_image("calibration_all_models_test.png", "Reliability diagram (test)")

    st.markdown("#### Research experiments")
    tabs = st.tabs(["Validation (selection)", "Feature sets", "Imbalance", "Tuning impact", "Calibration", "Ablation",
                    "Error analysis", "Failure modes"])
    tables = [("validation_results.csv", None), ("exp2_feature_sets.csv", None), ("exp3_imbalance.csv", None),
              ("exp4_tuning_impact.csv", None), ("exp5_calibration.csv", "calibration_methods_validation.png"),
              ("ablation_study.csv", None), ("exp7_recall_by_failure_mode.csv", None),
              ("failure_mode_results.csv", None)]
    for tab, (csv, img) in zip(tabs, tables):
        with tab:
            t = read_csv(csv)
            if t is not None:
                st.dataframe(t, hide_index=True, width="stretch")
            if csv == "exp2_feature_sets.csv" and t is not None:
                fig = px.bar(t, x="model", y="val_pr_auc", color="feature_set", barmode="group",
                             title="Validation PR-AUC: baseline vs engineered", text=t["val_pr_auc"].round(3))
                st.plotly_chart(style_fig(fig, 380), width="stretch")
            if csv == "ablation_study.csv" and t is not None:
                tt = t.iloc[1:].sort_values("delta_val_pr_auc")
                fig = px.bar(tt, x="delta_val_pr_auc", y="dropped_feature", orientation="h",
                             title="Change in validation PR-AUC when a feature is removed",
                             color_discrete_sequence=[CYAN])
                st.plotly_chart(style_fig(fig, 400), width="stretch")
            if csv == "exp7_recall_by_failure_mode.csv":
                prof = read_csv("exp7_outcome_profiles.csv")
                if prof is not None:
                    st.markdown("Mean values by outcome (TP / TN / FP / FN):")
                    st.dataframe(prof, hide_index=True, width="stretch")
            if img:
                fig_image(img)


# ============================================================================= PAGE 5
def page_xai():
    hero("Explainable AI", "SHAP (SHapley Additive exPlanations) attributes each prediction to the inputs. Global "
                           "views summarise the model; local views explain a single machine.")
    gi = read_csv("shap_global_importance.csv")
    meta = read_json("exp6_shap.json")
    if gi is None or meta is None or not need_artifacts():
        if gi is None:
            st.error("Run `python train.py` to compute SHAP explanations.")
        return
    st.info(f"Explaining **{meta['model']}** on {meta['n_explained']} test machines, in **{meta['output_space']}** space. "
            "SHAP describes how the model uses inputs — it is not proof of a physical cause. "
            "If the deployed model is calibrated, the explanation refers to the underlying uncalibrated model "
            "(calibration is monotonic, so rankings and directions are preserved).")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        g = gi.sort_values("mean_abs_shap")
        fig = px.bar(g, x="mean_abs_shap", y="label", orientation="h", text=g["share_pct"].map(lambda v: f"{v:.1f}%"),
                     title="Global feature importance (mean |SHAP|)", color_discrete_sequence=[CYAN])
        st.plotly_chart(style_fig(fig, 430), width="stretch")
    with c2:
        fig_image("shap_summary.png", "SHAP summary (beeswarm): colour = feature value, x = impact on output")
    fig_image("shap_dependence.png", "SHAP dependence plots for the top features")

    st.markdown("#### Individual prediction explanations")
    npz_path = REPORTS / "shap_values_test_sample.npz"
    if not npz_path.exists():
        return
    z = np.load(npz_path, allow_pickle=False)
    names = [str(n) for n in z["feature_names"]]
    thr = get_predictor().threshold
    y_true, proba = z["y_true"], z["proba"]
    outcome = np.where((y_true == 1) & (proba >= thr), "True positive",
                       np.where((y_true == 0) & (proba < thr), "True negative",
                                np.where(proba >= thr, "False positive", "False negative")))
    c1, c2 = st.columns([1, 2])
    with c1:
        kind = st.selectbox("Pick a case type", ["True positive", "False negative", "False positive", "True negative"])
        idx = np.where(outcome == kind)[0]
        if len(idx) == 0:
            st.info(f"No {kind.lower()}s in the explained sample.")
            return
        idx = idx[np.argsort(-proba[idx])]
        pos = st.slider("Case", 1, len(idx), 1) - 1
        i = int(idx[pos])
        row = dict(zip(names, z["data"][i]))
        st.markdown(kpi("Actual / predicted", f"{'Failure' if y_true[i] else 'No Failure'} / "
                        f"{'Failure' if proba[i] >= thr else 'No Failure'}", f"p = {proba[i]:.4f}"),
                    unsafe_allow_html=True)
        tdf = pd.DataFrame({"feature": [PRETTY.get(n, n) for n in names],
                            "value": [{0: "L", 1: "M", 2: "H"}.get(int(v), v) if n == "type" else round(float(v), 3)
                                      for n, v in row.items()],
                            "SHAP": np.round(z["values"][i], 4)})
        st.dataframe(tdf.sort_values("SHAP", key=np.abs, ascending=False), hide_index=True, width="stretch")
    with c2:
        st.plotly_chart(shap_waterfall_fig({"values": z["values"][i].tolist(), "base_value": float(z["base_values"][i]),
                                            "feature_names": names, "output_space": meta["output_space"]}),
                        width="stretch")


# ============================================================================= PAGE 6
def page_eda():
    hero("Dataset Analytics", "Interactive exploration of the AI4I 2020 Predictive Maintenance dataset (10,000 "
                              "synthetic machine records).")
    df = get_dataset().copy()
    df["Status"] = df[TARGET].map({0: "No Failure", 1: "Failure"})
    num = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min", "temp_diff_k",
           "power_w", "wear_torque_min_nm"]
    label = lambda c: PRETTY.get(c, c)  # noqa: E731
    kpi_row([("Rows", f"{len(df):,}", ""), ("Columns (raw)", "14", ""),
             ("Missing values", f"{int(df[INPUT_COLUMNS + [TARGET]].isna().sum().sum())}", ""),
             ("Duplicate feature rows", f"{int(df.duplicated(subset=INPUT_COLUMNS).sum())}", ""),
             ("Product types", "L / M / H", "")])
    st.write("")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distributions", "Relationships", "Correlation", "Failure analysis", "Data"])
    with tab1:
        c1, c2 = st.columns([1, 3])
        f = c1.selectbox("Feature", num, format_func=label)
        norm = c1.toggle("Normalise per class", value=True)
        fig = px.histogram(df, x=f, color="Status", nbins=60, barmode="overlay", opacity=.7,
                           histnorm="probability density" if norm else None,
                           color_discrete_map={"No Failure": BLUE, "Failure": RED}, title=f"Distribution of {label(f)}")
        c2.plotly_chart(style_fig(fig, 400), width="stretch")
        fig = px.box(df, x="Status", y=f, color="Status", color_discrete_map={"No Failure": BLUE, "Failure": RED},
                     points="outliers", title=f"{label(f)} by failure status")
        c2.plotly_chart(style_fig(fig, 380), width="stretch")
    with tab2:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", num, index=2, format_func=label)
        yv = c2.selectbox("Y axis", num, index=3, format_func=label)
        only_fail = c3.toggle("Highlight failures only", value=False)
        d = df if not only_fail else df[df[TARGET] == 1]
        fig = px.scatter(d.sort_values(TARGET), x=x, y=yv, color="Status", opacity=.6,
                         color_discrete_map={"No Failure": BLUE, "Failure": RED},
                         hover_data=["type", "tool_wear_min"], title=f"{label(yv)} vs {label(x)}")
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(style_fig(fig, 520), width="stretch")
    with tab3:
        cols = num + [TARGET] + FAILURE_MODE_COLUMNS
        corr = df[cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        x=[label(c) for c in cols], y=[label(c) for c in cols], title="Pearson correlation")
        st.plotly_chart(style_fig(fig, 640), width="stretch")
    with tab4:
        c1, c2 = st.columns(2)
        t = df.groupby("type")[TARGET].agg(["mean", "count"]).reindex(["L", "M", "H"]).reset_index()
        t["failure_rate_pct"] = 100 * t["mean"]
        fig = px.bar(t, x="type", y="failure_rate_pct", text=t["failure_rate_pct"].map(lambda v: f"{v:.2f}%"),
                     title="Failure rate by product type", color_discrete_sequence=[CYAN])
        c1.plotly_chart(style_fig(fig, 360), width="stretch")
        b = pd.cut(df["tool_wear_min"], bins=range(0, 280, 20), right=False)
        w = df.groupby(b, observed=False)[TARGET].mean().reset_index()
        w["bin"] = w["tool_wear_min"].astype(str); w["rate"] = 100 * w[TARGET]
        fig = px.bar(w, x="bin", y="rate", title="Failure rate by tool-wear bin [%]", color_discrete_sequence=[BLUE])
        c2.plotly_chart(style_fig(fig, 360), width="stretch")
        ov = read_csv("failure_mode_overlap.csv")
        if ov is not None:
            ov = ov.set_index(ov.columns[0])
            fig = px.imshow(ov, text_auto=True, color_continuous_scale=[[0, "#0B1C38"], [1, CYAN]],
                            title="Failure-mode co-occurrence (diagonal = count of each mode)")
            st.plotly_chart(style_fig(fig, 430), width="stretch")
        comp = df.groupby("Status")[num].mean().T
        comp["ratio (failure / normal)"] = comp["Failure"] / comp["No Failure"]
        st.markdown("**Mean feature values by status**")
        st.dataframe(comp.rename(index=label).round(3), width="stretch")
    with tab5:
        st.dataframe(df.head(500), width="stretch", height=380)
        st.markdown("**Descriptive statistics**")
        st.dataframe(df[num].describe().T.rename(index=label).round(3), width="stretch")


{"Overview": page_overview, "Machine Failure Prediction": page_prediction, "Batch Prediction": page_batch,
 "Model Performance": page_performance, "Explainable AI": page_xai, "Dataset Analytics": page_eda}[page]()
