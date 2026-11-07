"""Rule-based, ADVISORY maintenance recommendations.

Rules combine (a) the model's risk category, (b) the input operating conditions checked
against the failure-mechanism limits documented with the AI4I 2020 dataset, and (c) the
features SHAP marks as the strongest risk drivers. They are heuristics for planning
inspections — they do not guarantee that a failure will or will not occur.
"""
from __future__ import annotations

import math

DISCLAIMER = ("Advisory only: generated from documented rules and model output on a synthetic benchmark dataset. "
              "It does not guarantee failure prevention. Confirm with qualified maintenance staff and real sensor data.")


def operating_checks(inputs: dict, rules: dict) -> list[dict]:
    """Compare inputs with documented failure-mechanism limits. Returns triggered checks."""
    out = []
    wear = float(inputs["tool_wear_min"]); torque = float(inputs["torque_nm"])
    rpm = float(inputs["rotational_speed_rpm"])
    dT = float(inputs["process_temp_k"]) - float(inputs["air_temp_k"])
    power = torque * rpm * 2 * math.pi / 60
    strain = wear * torque
    limit = rules["overstrain_limit_min_nm"].get(str(inputs["type"]), 12000)

    if wear >= rules["tool_wear_critical_min"]:
        out.append({"mechanism": "Tool wear", "severity": "critical",
                    "detail": f"Tool wear {wear:.0f} min ≥ {rules['tool_wear_critical_min']} min (tool-wear failure zone)."})
    elif wear >= rules["tool_wear_warning_min"]:
        out.append({"mechanism": "Tool wear", "severity": "warning",
                    "detail": f"Tool wear {wear:.0f} min is approaching the {rules['tool_wear_critical_min']} min zone."})
    if dT < rules["heat_dissipation_temp_diff_k"] and rpm < rules["heat_dissipation_rpm"]:
        out.append({"mechanism": "Heat dissipation", "severity": "critical",
                    "detail": f"Temperature difference {dT:.1f} K < {rules['heat_dissipation_temp_diff_k']} K at "
                              f"{rpm:.0f} rpm < {rules['heat_dissipation_rpm']} rpm."})
    elif dT < rules["heat_dissipation_temp_diff_k"]:
        out.append({"mechanism": "Heat dissipation", "severity": "warning",
                    "detail": f"Low temperature difference ({dT:.1f} K); risk rises if speed drops below "
                              f"{rules['heat_dissipation_rpm']} rpm."})
    if power < rules["power_min_w"] or power > rules["power_max_w"]:
        out.append({"mechanism": "Power", "severity": "critical",
                    "detail": f"Mechanical power {power:.0f} W outside {rules['power_min_w']}–{rules['power_max_w']} W."})
    if strain > limit:
        out.append({"mechanism": "Overstrain", "severity": "critical",
                    "detail": f"Wear × torque {strain:.0f} min·Nm exceeds the type-{inputs['type']} limit {limit}."})
    elif strain > rules["overstrain_warning_ratio"] * limit:
        out.append({"mechanism": "Overstrain", "severity": "warning",
                    "detail": f"Wear × torque {strain:.0f} min·Nm is above {int(100 * rules['overstrain_warning_ratio'])}% "
                              f"of the type-{inputs['type']} limit {limit}."})
    return out


ACTIONS = {
    "Tool wear": "Schedule tool inspection/replacement at the next planned stop.",
    "Heat dissipation": "Inspect cooling and ventilation; avoid low-speed operation until the temperature gap recovers.",
    "Power": "Review the load profile: bring torque/speed back into the normal power band; check drive and power supply.",
    "Overstrain": "Reduce torque load or replace the tool earlier to limit cumulative strain.",
}
DRIVER_HINTS = {
    "tool_wear_min": "Tool wear", "wear_torque_min_nm": "Overstrain", "temp_diff_k": "Heat dissipation",
    "air_temp_k": "Heat dissipation", "process_temp_k": "Heat dissipation", "temp_diff_x_torque": "Heat dissipation",
    "power_w": "Power", "torque_nm": "Power", "rotational_speed_rpm": "Power",
}


def generate_recommendation(inputs: dict, probability: float, risk_category: str, top_features: list[dict],
                            rules: dict) -> dict:
    checks = operating_checks(inputs, rules)
    mechanisms = [c["mechanism"] for c in checks]
    # If no documented rule fired but the model still sees elevated risk, fall back to the
    # strongest positive SHAP driver to suggest where to look first.
    if not checks and risk_category != "Low":
        for f in top_features or []:
            if f.get("shap_value", 0) > 0 and f["feature"] in DRIVER_HINTS:
                mechanisms.append(DRIVER_HINTS[f["feature"]])
                break

    if risk_category == "High":
        priority = "Immediate attention"
        headline = "High predicted failure risk — plan an inspection as soon as operationally possible."
    elif risk_category == "Medium":
        priority = "Monitor closely"
        headline = "Elevated predicted risk — increase monitoring and include this machine in the next maintenance window."
    else:
        priority = "Routine"
        headline = "Low predicted risk — continue normal operation and routine maintenance."

    actions = [ACTIONS[m] for m in dict.fromkeys(mechanisms) if m in ACTIONS]
    if risk_category == "Low" and not checks:
        actions = ["No rule-based warnings triggered. Keep routine inspection intervals."]
    elif not actions:
        actions = ["Model risk is elevated without a single dominant rule; perform a general inspection."]
    return {"priority": priority, "headline": headline, "actions": actions, "rule_checks": checks,
            "probability": float(probability), "disclaimer": DISCLAIMER}
