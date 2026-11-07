"""PredictX AI — command-line inference.

Single machine:
    python predict.py --type M --air-temp 300 --process-temp 310 --rpm 1500 --torque 40 --tool-wear 120

Batch CSV:
    python predict.py --csv data/processed/test.csv --out outputs/predictions/test_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.inference import PredictXPredictor
from src.utils import InputValidationError, ModelNotFoundError, load_config, resolve_path


def format_single(res: dict) -> str:
    lines = [
        "=" * 64, "PredictX AI — machine failure risk", "=" * 64,
        f"Model                : {res['model_name']}",
        f"Predicted class      : {res['predicted_class']}",
        f"Failure probability  : {res['failure_probability']:.4f}  ({100 * res['failure_probability']:.2f}%)",
        f"Risk category        : {res['risk_category']}  "
        f"(Low < {res['risk_thresholds']['low_max']:.2f} <= Medium < {res['risk_thresholds']['high_min']:.3f} <= High)",
        "", "Most influential inputs (SHAP, model attribution — not proven physical causes):",
    ]
    for f in res["top_features"]:
        v = f["value"] if isinstance(f["value"], str) else f"{f['value']:.2f}"
        lines.append(f"  - {f['label']:<28} = {v:>10}   SHAP {f['shap_value']:+.3f}  ({f['direction']})")
    rec = res["recommendation"]
    lines += ["", f"Maintenance priority : {rec['priority']}", f"  {rec['headline']}"]
    lines += [f"  * {a}" for a in rec["actions"]]
    for c in rec["rule_checks"]:
        lines.append(f"  [rule:{c['severity']}] {c['detail']}")
    if res.get("failure_modes"):
        lines += ["", "Optional failure-mode module (independent per-mode probabilities):"]
        lines += [f"  - {k:<26} {v:.4f}" for k, v in res["failure_modes"].items()]
    if res["warnings"]:
        lines += ["", "Warnings:"] + [f"  ! {w}" for w in res["warnings"]]
    lines += ["", rec["disclaimer"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PredictX AI inference")
    ap.add_argument("--model", default=None, help="path to predictx_best_model.joblib")
    ap.add_argument("--csv", help="batch CSV input")
    ap.add_argument("--out", help="batch output CSV path")
    ap.add_argument("--type", dest="type", choices=["L", "M", "H"])
    ap.add_argument("--air-temp", type=float, help="Air temperature [K]")
    ap.add_argument("--process-temp", type=float, help="Process temperature [K]")
    ap.add_argument("--rpm", type=float, help="Rotational speed [rpm]")
    ap.add_argument("--torque", type=float, help="Torque [Nm]")
    ap.add_argument("--tool-wear", type=float, help="Tool wear [min]")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--no-explain", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    try:
        predictor = PredictXPredictor(args.model, cfg)
        if args.csv:
            df = pd.read_csv(args.csv)
            out = predictor.predict_batch(df, explain=not args.no_explain)
            path = Path(args.out) if args.out else resolve_path(cfg["paths"]["predictions_dir"]) / (
                Path(args.csv).stem + "_predictions.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
            valid = out[out["validation_error"] == ""]
            print(f"Scored {len(valid)}/{len(out)} rows -> {path}")
            print(valid["risk_category"].value_counts().to_string())
            return 0
        needed = [args.type, args.air_temp, args.process_temp, args.rpm, args.torque, args.tool_wear]
        if any(v is None for v in needed):
            ap.error("single prediction needs --type --air-temp --process-temp --rpm --torque --tool-wear (or use --csv)")
        res = predictor.predict({"type": args.type, "air_temp_k": args.air_temp, "process_temp_k": args.process_temp,
                                 "rotational_speed_rpm": args.rpm, "torque_nm": args.torque,
                                 "tool_wear_min": args.tool_wear}, explain=not args.no_explain)
        print(json.dumps(res, indent=2, default=str) if args.json else format_single(res))
        return 0
    except (InputValidationError, ModelNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
