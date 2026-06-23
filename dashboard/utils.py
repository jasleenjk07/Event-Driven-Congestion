from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "high_congestion_risk_model.pkl"
DURATION_MODEL_PATH = PROJECT_ROOT / "models" / "duration_risk_model.pkl"
DURATION_MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "duration_dataset.csv"
DURATION_TRUSTWORTHY_CAUSES = ["accident", "tree_fall", "congestion"]

HOLDOUT_FRACTION = 0.2

DEMO_SCENARIOS: dict[str, dict | None] = {
    "Custom (manual entry)": None,
    "Protest on ORR East — evening peak": {
        "event_type": "unplanned",
        "event_cause": "protest",
        "priority": "High",
        "requires_road_closure": True,
        "corridor": "ORR East 1",
        "zone": "East Zone 1",
        "year": 2024,
        "month": 3,
        "day": 15,
        "hour": 17,
        "latitude": 12.921876,
        "longitude": 77.645159,
        "use_location": True,
    },
    "VIP movement on Bellary Road": {
        "event_type": "planned",
        "event_cause": "vip_movement",
        "priority": "High",
        "requires_road_closure": True,
        "corridor": "Bellary Road 1",
        "zone": "North Zone 1",
        "year": 2024,
        "month": 6,
        "day": 10,
        "hour": 9,
        "latitude": 13.050000,
        "longitude": 77.590000,
        "use_location": True,
    },
    "Tree fall with road closure": {
        "event_type": "unplanned",
        "event_cause": "tree_fall",
        "priority": "Low",
        "requires_road_closure": True,
        "corridor": "Non-corridor",
        "zone": "Central Zone 2",
        "year": 2024,
        "month": 3,
        "day": 7,
        "hour": 17,
        "latitude": 13.006147,
        "longitude": 77.579435,
        "use_location": True,
    },
}
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
RECOMMENDATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "recommendation_dataset.csv"

MODEL_FEATURES = [
    "event_type",
    "event_cause",
    "priority",
    "requires_road_closure", 
    "hour",
    "month",
    "is_weekend",
    "corridor",
    "zone",
]

PRIORITY_SCORE = {"Low": 1, "Medium": 2, "High": 3}

CAUSE_SCORE = {
    "vip_movement": 5,
    "procession": 5,
    "public_event": 4,
    "protest": 5,
    "construction": 4,
    "congestion": 4,
    "accident": 4,
    "vehicle_breakdown": 2,
    "water_logging": 4,
    "tree_fall": 3,
    "pot_holes": 2,
    "others": 1,
    "debris": 2,
    "fog / low visibility": 3,
}

EVENT_CAUSE_LABELS = {
    "accident": "Accident",
    "congestion": "Congestion",
    "construction": "Construction",
    "debris": "Debris",
    "fog / low visibility": "Fog / Low Visibility",
    "others": "Others",
    "pot_holes": "Pot Holes",
    "procession": "Procession",
    "protest": "Protest",
    "public_event": "Public Event",
    "road_conditions": "Road Conditions",
    "test_demo": "Test / Demo",
    "tree_fall": "Tree Fall",
    "vehicle_breakdown": "Vehicle Breakdown",
    "vip_movement": "VIP Movement",
    "water_logging": "Water Logging",
}


def format_event_cause(value: str) -> str:
    return EVENT_CAUSE_LABELS.get(value, value.replace("_", " ").title())


def get_time_score(hour: int) -> int:
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return 2
    return 0


def compute_eci(
    priority: str,
    event_cause: str,
    requires_road_closure: bool,
    hour: int,
    is_weekend: int,
) -> dict:
    priority_score = PRIORITY_SCORE.get(priority, 1)
    cause_score = CAUSE_SCORE.get(event_cause, 2)
    closure_score = 3 if requires_road_closure else 0
    time_score = get_time_score(hour)
    weekend_score = 1 if is_weekend else 0
    eci = priority_score + cause_score + closure_score + time_score + weekend_score

    if eci >= 10:
        level = "High"
    elif eci >= 6:
        level = "Medium"
    else:
        level = "Low"

    return {
        "priority_score": priority_score,
        "cause_score": cause_score,
        "closure_score": closure_score,
        "time_score": time_score,
        "weekend_score": weekend_score,
        "eci": eci,
        "congestion_risk_level": level,
    }


def recommend_resources(prediction: int, event_cause: str, priority: str) -> dict:
    if prediction == 0:
        officers = 4 if priority == "High" else 2
        return {
            "officers": officers,
            "barricades": 0,
            "diversion": "Not Required",
            "monitoring": "No",
        }

    officers = 10
    barricades = 6
    diversion = "Optional"

    if event_cause == "vip_movement":
        officers, barricades, diversion = 20, 10, "Required"
    elif event_cause == "procession":
        officers, barricades, diversion = 15, 8, "Required"
    elif event_cause == "protest":
        officers, barricades, diversion = 18, 10, "Required"
    elif event_cause == "public_event":
        officers, barricades, diversion = 12, 8, "Required"
    elif event_cause == "construction":
        officers, barricades, diversion = 8, 12, "Required"
    elif event_cause == "water_logging":
        officers, barricades, diversion = 6, 10, "Required"
    elif event_cause == "accident":
        officers, barricades = 8, 4
    elif event_cause == "tree_fall":
        officers, barricades = 6, 6
    elif event_cause == "vehicle_breakdown":
        officers, barricades = 4, 2

    if priority == "High":
        officers += 2

    return {
        "officers": officers,
        "barricades": barricades,
        "diversion": diversion,
        "monitoring": "Yes",
    }


def build_feature_row(
    event_type: str,
    event_cause: str,
    priority: str,
    requires_road_closure: bool,  
    hour: int,
    month: int,
    is_weekend: int,
    corridor: str,
    zone: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_type": event_type,
                "event_cause": event_cause,
                "priority": priority,
                "requires_road_closure": int(requires_road_closure), 
                "hour": hour,
                "month": month,
                "is_weekend": is_weekend,
                "corridor": corridor,
                "zone": zone,
            }
        ]
    )


def predict_congestion(model, feature_row: pd.DataFrame) -> tuple[int, float]:
    prediction = int(model.predict(feature_row)[0])
    probability = float(model.predict_proba(feature_row)[0][1])
    return prediction, probability


def risk_label(prediction: int) -> str:
    return "HIGH" if prediction == 1 else "NORMAL"


def risk_color(prediction: int) -> str:
    return "#dc2626" if prediction == 1 else "#16a34a"


def eci_color(level: str) -> str:
    return {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"}.get(level, "#64748b")


def load_option_values(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "event_type": sorted(df["event_type"].dropna().unique().tolist()),
        "event_cause": sorted(df["event_cause"].dropna().unique().tolist()),
        "priority": sorted(df["priority"].dropna().unique().tolist()),
        "corridor": sorted(df["corridor"].dropna().unique().tolist()),
        "zone": sorted(df["zone"].dropna().unique().tolist()),
    }


def ml_eci_aligned(prediction: int, eci_level: str) -> bool:
    ml_high = prediction == 1
    eci_high = eci_level == "High"
    return ml_high == eci_high


def get_time_based_holdout_df(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["start_datetime"] = pd.to_datetime(ordered["start_datetime"], utc=True)
    ordered = ordered.sort_values("start_datetime").reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - HOLDOUT_FRACTION))
    return ordered.iloc[split_idx:].copy()


def evaluate_holdout_predictions(model, df: pd.DataFrame) -> dict:
    holdout = get_time_based_holdout_df(df)
    result = evaluate_predictions(model, holdout)
    result["holdout_size"] = len(holdout)
    result["holdout_start"] = holdout["start_datetime"].min()
    result["holdout_end"] = holdout["start_datetime"].max()
    return result


def get_scenario_defaults(scenario_name: str) -> dict:
    scenario = DEMO_SCENARIOS.get(scenario_name)
    if scenario is None:
        return {
            "event_type": "unplanned",
            "event_cause": "protest",
            "priority": "High",
            "requires_road_closure": False,
            "corridor": "ORR East 1",
            "zone": "East Zone 1",
            "year": 2024,
            "month": 3,
            "day": 15,
            "hour": 12,
            "latitude": 12.9716,
            "longitude": 77.5946,
            "use_location": False,
        }
    return scenario


def pick_index(options: list[str], value: str, fallback: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return fallback


def build_deployment_plan(
    event_type: str,
    event_cause: str,
    priority: str,
    requires_road_closure: bool,
    corridor: str,
    zone: str,
    event_datetime: datetime | pd.Timestamp,
    prediction: int,
    probability: float,
    eci_data: dict,
    recs: dict,
    duration_pred: int | None = None,
    duration_prob: float | None = None,
) -> pd.DataFrame:
    if hasattr(event_datetime, "to_pydatetime"):
        event_datetime = event_datetime.to_pydatetime()

    row = {
        "event_type": event_type,
        "event_cause": format_event_cause(event_cause),
        "priority": priority,
        "requires_road_closure": requires_road_closure,
        "corridor": corridor,
        "zone": zone,
        "event_start": event_datetime.isoformat() if isinstance(event_datetime, datetime) else str(event_datetime),
        "congestion_risk": risk_label(prediction),
        "risk_probability_pct": round(probability * 100, 1),
        "eci": eci_data["eci"],
        "eci_risk_level": eci_data["congestion_risk_level"],
        "recommended_officers": recs["officers"],
        "recommended_barricades": recs["barricades"],
        "diversion_plan": recs["diversion"],
        "monitoring": recs["monitoring"],
    }
    if duration_pred is not None and duration_prob is not None:
        row["duration_risk"] = "LONG" if duration_pred == 1 else "TYPICAL"
        row["duration_probability_pct"] = round(duration_prob * 100, 1)
    return pd.DataFrame([row])


def evaluate_predictions(model, df: pd.DataFrame) -> dict:
    features = df[MODEL_FEATURES].copy()
    features["requires_road_closure"] = features["requires_road_closure"].astype(int)
    features[["hour", "month", "is_weekend"]] = features[["hour", "month", "is_weekend"]].astype(int)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    actual = df["high_congestion_risk"].astype(int).values

    ml_high_eci_not = int(((predictions == 1) & (df["congestion_risk_level"] != "High")).sum())
    ml_normal_eci_high = int(((predictions == 0) & (df["congestion_risk_level"] == "High")).sum())

    return {
        "total_events": len(df),
        "accuracy": float((predictions == actual).mean()),
        "precision_high": float(((predictions == 1) & (actual == 1)).sum() / max((predictions == 1).sum(), 1)),
        "recall_high": float(((predictions == 1) & (actual == 1)).sum() / max((actual == 1).sum(), 1)),
        "false_positives": int(((predictions == 1) & (actual == 0)).sum()),
        "false_negatives": int(((predictions == 0) & (actual == 1)).sum()),
        "ml_high_eci_not_high": ml_high_eci_not,
        "ml_normal_eci_high": ml_normal_eci_high,
        "predictions": predictions,
        "probabilities": probabilities,
        "actual": actual,
    }

def predict_duration_risk(model, feature_row: pd.DataFrame) -> tuple[int, float]:
    prediction = int(model.predict(feature_row)[0])
    probability = float(model.predict_proba(feature_row)[0][1])
    return prediction, probability
