#!/usr/bin/env python3
"""Export key figures for hackathon submission into reports/figures/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_DATA = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
REC_DATA = PROJECT_ROOT / "data" / "processed" / "recommendation_dataset.csv"

sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))
from utils import format_event_cause, get_time_based_holdout_df  # noqa: E402

sns.set_theme(style="whitegrid")


def save_fig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def main() -> None:
    model_df = pd.read_csv(MODEL_DATA)
    rec_df = pd.read_csv(REC_DATA)

    # Events by cause
    cause_counts = (
        model_df["event_cause"].map(format_event_cause).value_counts().head(10).sort_values()
    )
    plt.figure(figsize=(10, 6))
    cause_counts.plot(kind="barh", color="#2563eb")
    plt.title("Top Event Causes — Bengaluru ASTraM Data")
    plt.xlabel("Event count")
    save_fig("events_by_cause.png")

    # ECI distribution
    order = ["Low", "Medium", "High"]
    eci_counts = model_df["congestion_risk_level"].value_counts().reindex(order, fill_value=0)
    plt.figure(figsize=(8, 5))
    eci_counts.plot(kind="bar", color=["#16a34a", "#f59e0b", "#dc2626"])
    plt.title("Event Congestion Index (ECI) Risk Levels")
    plt.ylabel("Events")
    plt.xticks(rotation=0)
    save_fig("eci_distribution.png")

    # Hourly pattern
    hourly = model_df.groupby("hour").size()
    plt.figure(figsize=(10, 5))
    hourly.plot(color="#7c3aed")
    plt.title("Events by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Events")
    save_fig("events_by_hour.png")

    # High-risk by zone
    zone_high = rec_df[rec_df["prediction"] == 1]["zone"].value_counts().head(8).sort_values()
    plt.figure(figsize=(10, 6))
    zone_high.plot(kind="barh", color="#dc2626")
    plt.title("High-Risk ML Predictions by Zone")
    plt.xlabel("Events")
    save_fig("high_risk_by_zone.png")

    # High-risk by corridor
    corridor_high = rec_df[rec_df["prediction"] == 1]["corridor"].value_counts().head(10).sort_values()
    plt.figure(figsize=(10, 7))
    corridor_high.plot(kind="barh", color="#ea580c")
    plt.title("High-Risk ML Predictions by Corridor")
    plt.xlabel("Events")
    save_fig("high_risk_by_corridor.png")

    # Holdout vs full timeline
    holdout = get_time_based_holdout_df(model_df)
    plt.figure(figsize=(10, 4))
    plt.axvline(
        holdout["start_datetime"].min(),
        color="#dc2626",
        linestyle="--",
        label="Holdout start (most recent 20%)",
    )
    monthly = model_df.copy()
    monthly["start_datetime"] = pd.to_datetime(monthly["start_datetime"], utc=True)
    monthly.groupby(monthly["start_datetime"].dt.to_period("M")).size().astype(int).plot(
        color="#2563eb", label="Events per month"
    )
    plt.title("Train / Holdout Split (time-based)")
    plt.ylabel("Events")
    plt.legend()
    save_fig("time_based_split.png")

    print(f"Done — {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
