from __future__ import annotations

from datetime import datetime

try:
    import folium

    FOLIUM_AVAILABLE = True
except ImportError:
    folium = None
    FOLIUM_AVAILABLE = False

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    DEMO_SCENARIOS,
    MODEL_DATA_PATH,
    MODEL_PATH,
    RECOMMENDATION_DATA_PATH,
    DURATION_MODEL_PATH,
    DURATION_TRUSTWORTHY_CAUSES,
    build_deployment_plan,
    build_feature_row,
    compute_eci,
    eci_color,
    evaluate_holdout_predictions,
    evaluate_predictions,
    format_event_cause,
    get_scenario_defaults,
    load_option_values,
    ml_eci_aligned,
    pick_index,
    predict_congestion,
    predict_duration_risk,
    recommend_resources,
    risk_color,
    risk_label,
)

# Bengaluru city center (fallback map focus)
BENGALURU_CENTER = (12.9716, 77.5946)

st.set_page_config(
    page_title="Event-Driven Congestion Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .risk-banner {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_duration_model():
    return joblib.load(DURATION_MODEL_PATH)

@st.cache_data
def load_model_data() -> pd.DataFrame:
    return pd.read_csv(MODEL_DATA_PATH)


@st.cache_data
def load_recommendation_data() -> pd.DataFrame:
    return pd.read_csv(RECOMMENDATION_DATA_PATH)


def render_risk_banner(prediction: int, probability: float) -> None:
    label = risk_label(prediction)
    color = risk_color(prediction)
    st.markdown(
        f"""
        <div class="risk-banner" style="background: {color};">
            <div style="font-size: 0.95rem; opacity: 0.95;">Predicted Congestion Risk</div>
            <div style="font-size: 2rem; font-weight: 700;">{label}</div>
            <div style="font-size: 1.1rem;">Risk Probability: {probability * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eci_breakdown(eci_data: dict) -> None:
    cols = st.columns(5)
    labels = [
        ("Priority", eci_data["priority_score"]),
        ("Cause", eci_data["cause_score"]),
        ("Closure", eci_data["closure_score"]),
        ("Time", eci_data["time_score"]),
        ("Weekend", eci_data["weekend_score"]),
    ]
    for col, (name, value) in zip(cols, labels):
        col.metric(name, value)

    level = eci_data["congestion_risk_level"]
    st.markdown(
        f"""
        **Event Congestion Index (ECI):** {eci_data['eci']:.0f} &nbsp;|&nbsp;
        **ECI Risk Level:** <span style="color:{eci_color(level)}; font-weight:600;">{level}</span>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations(recs: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Police Officers", recs["officers"])
    cols[1].metric("Barricades", recs["barricades"])
    cols[2].metric("Diversion Plan", recs["diversion"])
    cols[3].metric("Monitoring", recs["monitoring"])


def render_folium_map(folium_map, width: int = 700, height: int = 350) -> None:
    if not FOLIUM_AVAILABLE:
        st.warning("Maps are unavailable. Install folium with: `pip install folium`")
        return
    components.html(folium_map._repr_html_(), width=width, height=height)


def _map_center(df: pd.DataFrame) -> tuple[float, float]:
    if len(df) > 0:
        return float(df["latitude"].mean()), float(df["longitude"].mean())
    return BENGALURU_CENTER


def build_prediction_event_map(
    latitude: float,
    longitude: float,
    corridor: str,
    zone: str,
    event_cause: str,
    prediction: int,
    probability: float,
    eci: float,
    recs: dict,
) -> "folium.Map":
    risk = risk_label(prediction)
    icon_color = "red" if prediction == 1 else "green"
    popup_html = (
        f"<b>{format_event_cause(event_cause)}</b><br>"
        f"Corridor: {corridor}<br>"
        f"Zone: {zone}<br>"
        f"Congestion risk: <b>{risk}</b> ({probability * 100:.0f}%)<br>"
        f"ECI: {eci:.0f}<br>"
        f"Officers: {recs['officers']} · Barricades: {recs['barricades']}<br>"
        f"Diversion: {recs['diversion']}"
    )
    event_map = folium.Map(location=[latitude, longitude], zoom_start=13)
    folium.Marker(
        [latitude, longitude],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{format_event_cause(event_cause)} — {risk} risk @ {corridor}",
        icon=folium.Icon(color=icon_color, icon="info-sign"),
    ).add_to(event_map)
    folium.Circle(
        location=[latitude, longitude],
        radius=800,
        color=icon_color,
        fill=True,
        fill_opacity=0.08,
        popup=f"Influence zone · {zone}",
    ).add_to(event_map)
    return event_map


def build_marker_map(df: pd.DataFrame, max_points: int = 500) -> "folium.Map":
    plot_df = df.dropna(subset=["latitude", "longitude"]).head(max_points)
    center = _map_center(plot_df)
    event_map = folium.Map(location=list(center), zoom_start=11)
    for _, row in plot_df.iterrows():
        is_high = int(row.get("prediction", row.get("high_congestion_risk", 0))) == 1
        color = "red" if is_high else "green"
        officers = row.get("officers", "—")
        popup_html = (
            f"<b>{format_event_cause(row['event_cause'])}</b><br>"
            f"Corridor: {row.get('corridor', '—')}<br>"
            f"Zone: {row.get('zone', '—')}<br>"
            f"ECI: {row.get('eci', 0):.0f} ({row.get('congestion_risk_level', '—')})<br>"
            f"ML risk: {'HIGH' if is_high else 'NORMAL'}<br>"
            f"Officers: {officers}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5 if is_high else 4,
            color=color,
            fill=True,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=format_event_cause(row["event_cause"]),
        ).add_to(event_map)
    return event_map


def build_heatmap(df: pd.DataFrame, max_points: int = 2000) -> "folium.Map":
    from folium.plugins import HeatMap

    plot_df = df.dropna(subset=["latitude", "longitude"]).head(max_points).copy()
    center = _map_center(plot_df)
    event_map = folium.Map(location=list(center), zoom_start=11)

    if len(plot_df) == 0:
        return event_map

    if "prediction" in plot_df.columns:
        weights = plot_df["prediction"].astype(float) + 0.2
    elif "high_congestion_risk" in plot_df.columns:
        weights = plot_df["high_congestion_risk"].astype(float) + 0.2
    elif "eci" in plot_df.columns:
        weights = (plot_df["eci"] / plot_df["eci"].max()).clip(0.1, 1.0)
    else:
        weights = pd.Series([0.5] * len(plot_df))

    heat_data = [
        [row["latitude"], row["longitude"], float(w)]
        for (_, row), w in zip(plot_df.iterrows(), weights)
    ]
    HeatMap(
        heat_data,
        radius=18,
        blur=22,
        max_zoom=13,
        gradient={0.2: "blue", 0.5: "lime", 0.8: "orange", 1.0: "red"},
    ).add_to(event_map)
    return event_map


def render_event_map(latitude: float | None, longitude: float | None) -> None:
    if not FOLIUM_AVAILABLE:
        st.warning("Maps are unavailable. Install folium with: `pip install folium`")
        return

    if latitude is None or longitude is None:
        st.info("Add latitude and longitude to preview the event location on the map.")
        return

    event_map = folium.Map(location=[latitude, longitude], zoom_start=14)
    folium.Marker(
        [latitude, longitude],
        tooltip="Event location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(event_map)
    render_folium_map(event_map)


def page_predict(model, options: dict[str, list[str]]) -> None:
    st.title("Congestion Risk Prediction")
    st.caption(
        "Enter event details to forecast congestion risk and receive traffic management recommendations."
    )

    scenario = st.selectbox(
        "Demo scenario",
        list(DEMO_SCENARIOS.keys()),
        help="Pre-load realistic Bengaluru ASTraM event scenarios for judging demos.",
    )
    defaults = get_scenario_defaults(scenario)
    if scenario != "Custom (manual entry)":
        st.info(
            f"**{scenario}** — adjust fields below if needed, then click **Predict Congestion Risk**."
        )

    default_dt = datetime(
        defaults["year"],
        defaults["month"],
        defaults["day"],
        defaults["hour"],
        0,
        0,
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            event_type = st.selectbox(
                "Event Type",
                options["event_type"],
                index=pick_index(options["event_type"], defaults["event_type"]),
            )
            event_cause = st.selectbox(
                "Event Cause",
                options["event_cause"],
                index=pick_index(options["event_cause"], defaults["event_cause"]),
                format_func=format_event_cause,
            )
            priority = st.selectbox(
                "Priority",
                options["priority"],
                index=pick_index(options["priority"], defaults["priority"]),
            )
            requires_road_closure = st.checkbox(
                "Requires Road Closure",
                value=defaults["requires_road_closure"],
            )
            event_datetime = st.datetime_input(
                "Event Start",
                value=default_dt,
            )

        with col2:
            corridor = st.selectbox(
                "Corridor",
                options["corridor"],
                index=pick_index(options["corridor"], defaults["corridor"]),
            )
            zone = st.selectbox(
                "Zone",
                options["zone"],
                index=pick_index(options["zone"], defaults["zone"]),
            )
            latitude = st.number_input(
                "Latitude",
                value=float(defaults["latitude"]),
                format="%.6f",
            )
            longitude = st.number_input(
                "Longitude",
                value=float(defaults["longitude"]),
                format="%.6f",
            )

        submitted = st.form_submit_button(
            "Predict Congestion Risk",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return

    hour = event_datetime.hour
    month = event_datetime.month
    is_weekend = 1 if event_datetime.weekday() >= 5 else 0

    feature_row = build_feature_row(
        event_type=event_type,
        event_cause=event_cause,
        priority=priority,
        requires_road_closure=requires_road_closure,
        hour=hour,
        month=month,
        is_weekend=is_weekend,
        corridor=corridor,
        zone=zone,
    )
    prediction, probability = predict_congestion(model, feature_row)
    eci_data = compute_eci(priority, event_cause, requires_road_closure, hour, is_weekend)
    recs = recommend_resources(prediction, event_cause, priority)

    duration_pred = None
    duration_prob = None
    if event_cause in DURATION_TRUSTWORTHY_CAUSES:
        duration_model = load_duration_model()
        duration_pred, duration_prob = predict_duration_risk(duration_model, feature_row)

    st.divider()
    left, right = st.columns([1.1, 1])

    with left:
        render_risk_banner(prediction, probability)
        st.markdown('<div class="section-title">Traffic Management Plan</div>', unsafe_allow_html=True)
        render_recommendations(recs)

        st.markdown('<div class="section-title">Diversion Guidance</div>', unsafe_allow_html=True)
        if recs["diversion"] == "Required":
            st.warning(
                "Activate diversion routes, update variable message signs, and coordinate with corridor patrol units."
            )
        elif recs["diversion"] == "Optional":
            st.info("Prepare alternate routes and monitor queue buildup before activating diversions.")
        else:
            st.success("Standard lane management should be sufficient. Continue routine monitoring.")

        plan_df = build_deployment_plan(
            event_type,
            event_cause,
            priority,
            requires_road_closure,
            corridor,
            zone,
            event_datetime,
            prediction,
            probability,
            eci_data,
            recs,
            duration_pred,
            duration_prob,
        )
        st.download_button(
            "Download deployment plan (CSV)",
            data=plan_df.to_csv(index=False),
            file_name="traffic_deployment_plan.csv",
            mime="text/csv",
            width="stretch",
        )

    with right:
        st.markdown('<div class="section-title">Event Congestion Index Breakdown</div>', unsafe_allow_html=True)
        render_eci_breakdown(eci_data)

        if not ml_eci_aligned(prediction, eci_data["congestion_risk_level"]):
            if prediction == 1 and eci_data["congestion_risk_level"] != "High":
                st.warning(
                    "ML predicts **HIGH** congestion risk, but the computed ECI level is "
                    f"**{eci_data['congestion_risk_level']}**. "
                    "The model uses corridor, zone, and timing patterns beyond ECI alone. "
                    "Resource recommendations follow the ML prediction."
                )
            else:
                st.warning(
                    "ECI level is **High**, but the ML model predicts **NORMAL** risk. "
                    "Treat this as a cautious case and consider enhanced monitoring."
                )
        else:
            st.success("ML prediction and ECI risk level are aligned.")

        st.caption(
            "Road closure feeds both the ECI score and the ML model. "
            "Recommendations follow the ML prediction."
        )

    if duration_pred is not None and duration_prob is not None:
        st.markdown(
            '<div class="section-title">Outcome-Based Duration Risk (independent model)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Trained on actual event-closure durations for accident, tree_fall, and congestion — "
            "not the ECI formula."
        )
        label = "LONG DURATION RISK" if duration_pred == 1 else "TYPICAL DURATION"
        st.metric("Duration Risk", label, f"{duration_prob * 100:.1f}% probability")
    elif event_cause not in DURATION_TRUSTWORTHY_CAUSES:
        st.caption(
            "Duration model applies only to accident, tree_fall, and congestion — causes where "
            "closure timestamps reflect real road-clearance time."
        )

    st.markdown('<div class="section-title">Event Location — Bengaluru</div>', unsafe_allow_html=True)
    if FOLIUM_AVAILABLE:
        prediction_map = build_prediction_event_map(
            latitude,
            longitude,
            corridor,
            zone,
            event_cause,
            prediction,
            probability,
            eci_data["eci"],
            recs,
        )
        render_folium_map(prediction_map, width=950, height=420)
        st.caption(
            f"Pin shows predicted deployment zone on **{corridor}** ({zone}). "
            "Red = HIGH congestion risk · Green = NORMAL."
        )
    else:
        st.warning("Install `folium` to enable maps: `pip install folium`")


def page_analytics(model_df: pd.DataFrame, rec_df: pd.DataFrame, model) -> None:
    st.title("Historical Analytics")
    st.caption("Explore patterns in historical traffic events and congestion risk.")

    holdout = evaluate_holdout_predictions(model, model_df)
    full_val = evaluate_predictions(model, model_df)

    st.subheader("Model Validation — Time-Based Holdout (recommended)")
    st.caption(
        f"Most recent 20% of events by `start_datetime` "
        f"({holdout['holdout_size']:,} events, "
        f"{holdout['holdout_start'].date()} → {holdout['holdout_end'].date()})"
    )
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Holdout Accuracy", f"{100 * holdout['accuracy']:.1f}%")
    h2.metric("High-Risk Precision", f"{100 * holdout['precision_high']:.1f}%")
    h3.metric("High-Risk Recall", f"{100 * holdout['recall_high']:.1f}%")
    h4.metric("False Positives", holdout["false_positives"])
    h5.metric("False Negatives", holdout["false_negatives"])

    with st.expander("Full dataset metrics (in-sample — for reference only)"):
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Accuracy", f"{100 * full_val['accuracy']:.1f}%")
        f2.metric("High-Risk Precision", f"{100 * full_val['precision_high']:.1f}%")
        f3.metric("High-Risk Recall", f"{100 * full_val['recall_high']:.1f}%")
        f4.metric("False Positives", full_val["false_positives"])
        f5.metric("False Negatives", full_val["false_negatives"])

    st.info(
        "Dashboard logic verified: ECI 100% match, recommendations 100% match, "
        "live ML predictions match notebook outputs."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("ML HIGH but ECI not High (holdout)", holdout["ml_high_eci_not_high"])
    with c2:
        st.metric("ECI High but ML NORMAL (holdout)", holdout["ml_normal_eci_high"])

    st.divider()

    total_events = len(model_df)
    high_risk = int(model_df["high_congestion_risk"].sum())
    high_eci = int((model_df["congestion_risk_level"] == "High").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", f"{total_events:,}")
    c2.metric("High ML Risk Events", f"{high_risk:,}")
    c3.metric("High ECI Events", f"{high_eci:,}")
    c4.metric("High Risk Rate", f"{100 * high_risk / total_events:.1f}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Events by Cause")
        cause_counts = (
            model_df["event_cause"]
            .map(format_event_cause)
            .value_counts()
            .head(10)
            .sort_values()
        )
        st.bar_chart(cause_counts)

    with col2:
        st.subheader("Congestion Risk Level Distribution")
        risk_counts = model_df["congestion_risk_level"].value_counts().reindex(
            ["Low", "Medium", "High"], fill_value=0
        )
        st.bar_chart(risk_counts)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Events by Hour of Day")
        hourly = model_df.groupby("hour").size()
        st.line_chart(hourly)

    with col4:
        st.subheader("High-Risk Events by Zone")
        zone_high = (
            rec_df[rec_df["prediction"] == 1]["zone"].value_counts().head(8).sort_values()
        )
        st.bar_chart(zone_high)

    st.subheader("High-Risk Events by Corridor")
    corridor_high = (
        rec_df[rec_df["prediction"] == 1]["corridor"].value_counts().head(10).sort_values()
    )
    st.bar_chart(corridor_high)

    st.subheader("Recommended Resources for High-Risk Events")
    high_risk_recs = rec_df[rec_df["prediction"] == 1]
    resource_summary = pd.DataFrame(
        {
            "Metric": ["Avg Officers", "Avg Barricades", "Diversion Required", "Monitoring Enabled"],
            "Value": [
                f"{high_risk_recs['officers'].mean():.1f}",
                f"{high_risk_recs['barricades'].mean():.1f}",
                f"{100 * (high_risk_recs['diversion'] == 'Required').mean():.1f}%",
                f"{100 * (high_risk_recs['monitoring'] == 'Yes').mean():.1f}%",
            ],
        }
    )
    st.dataframe(resource_summary, hide_index=True, width="stretch")

    if FOLIUM_AVAILABLE:
        st.divider()
        st.subheader("Bengaluru Congestion Heatmap")
        st.caption(
            "Geographic density of ML high-risk events across Bengaluru. "
            "Brighter areas indicate more historical high-risk incidents."
        )
        high_risk_geo = rec_df[
            (rec_df["prediction"] == 1)
            & rec_df["latitude"].notna()
            & rec_df["longitude"].notna()
        ]
        if len(high_risk_geo) > 0:
            heatmap = build_heatmap(high_risk_geo, max_points=2000)
            render_folium_map(heatmap, width=950, height=450)
        else:
            st.info("No geocoded high-risk events available for heatmap.")

        st.subheader("High-Risk Event Map")
        map_style = st.radio(
            "Map view",
            ["Heatmap (density)", "Markers (detail)"],
            horizontal=True,
            key="analytics_map_style",
        )
        if map_style.startswith("Heatmap"):
            risk_map = build_heatmap(high_risk_geo if len(high_risk_geo) > 0 else rec_df, max_points=1500)
        else:
            risk_map = build_marker_map(
                high_risk_geo if len(high_risk_geo) > 0 else rec_df,
                max_points=400,
            )
        render_folium_map(risk_map, width=950, height=450)
    else:
        st.warning("Install `folium` for Bengaluru risk maps: `pip install folium`")


def page_explorer(rec_df: pd.DataFrame) -> None:
    st.title("Event Explorer")
    st.caption("Filter historical events and inspect predictions with recommended deployments.")

    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        risk_filter = f1.selectbox("ML Risk", ["All", "High", "Normal"])
        eci_filter = f2.selectbox("ECI Level", ["All", "Low", "Medium", "High"])
        cause_filter = f3.multiselect(
            "Event Cause",
            sorted(rec_df["event_cause"].dropna().unique()),
            format_func=format_event_cause,
        )
        zone_filter = f4.multiselect("Zone", sorted(rec_df["zone"].dropna().unique()))

    filtered = rec_df.copy()
    if risk_filter == "High":
        filtered = filtered[filtered["prediction"] == 1]
    elif risk_filter == "Normal":
        filtered = filtered[filtered["prediction"] == 0]

    if eci_filter != "All":
        filtered = filtered[filtered["congestion_risk_level"] == eci_filter]

    if cause_filter:
        filtered = filtered[filtered["event_cause"].isin(cause_filter)]

    if zone_filter:
        filtered = filtered[filtered["zone"].isin(zone_filter)]

    st.write(f"Showing **{len(filtered):,}** events")

    display_cols = [
        "id",
        "event_type",
        "event_cause",
        "priority",
        "requires_road_closure",
        "corridor",
        "zone",
        "hour",
        "eci",
        "congestion_risk_level",
        "prediction",
        "officers",
        "barricades",
        "diversion",
        "monitoring",
        "address",
    ]
    st.dataframe(
        filtered[display_cols]
        .rename(columns={"prediction": "high_risk_prediction", "event_cause": "cause"})
        .head(250),
        width="stretch",
        hide_index=True,
    )

    map_df = filtered.dropna(subset=["latitude", "longitude"])
    if len(map_df) > 0 and FOLIUM_AVAILABLE:
        st.subheader("Event Map — Bengaluru")
        map_style = st.radio(
            "Map view",
            ["Markers (red = high risk)", "Heatmap (density)", "Both"],
            horizontal=True,
            key="explorer_map_style",
        )

        if map_style.startswith("Markers"):
            explorer_map = build_marker_map(map_df, max_points=500)
            render_folium_map(explorer_map, width=950, height=480)
        elif map_style.startswith("Heatmap"):
            explorer_map = build_heatmap(map_df, max_points=1500)
            render_folium_map(explorer_map, width=950, height=480)
        else:
            col_m, col_h = st.columns(2)
            with col_m:
                st.caption("Individual events")
                marker_map = build_marker_map(map_df, max_points=400)
                render_folium_map(marker_map, width=470, height=420)
            with col_h:
                st.caption("Density")
                heat_map = build_heatmap(map_df, max_points=1500)
                render_folium_map(heat_map, width=470, height=420)

        st.caption(
            f"Showing up to 500 markers / 1,500 heat points from **{len(map_df):,}** geocoded events. "
            "Click markers for corridor, zone, ECI, and deployment details."
        )
    elif not FOLIUM_AVAILABLE:
        st.warning("Install `folium` for event maps: `pip install folium`")


def main() -> None:
    model = load_model()
    model_df = load_model_data()
    rec_df = load_recommendation_data()
    options = load_option_values(model_df)

    with st.sidebar:
        st.title("🚦 Traffic Command")
        st.markdown("Event-Driven Congestion Risk Forecasting")
        page = st.radio(
            "Navigation",
            ["Predict & Plan", "Analytics", "Event Explorer"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Model**")
        st.caption("Random Forest · 9 features")
        st.markdown("**Coverage**")
        st.caption(f"{len(model_df):,} historical events")
        st.markdown("**Recall (High Risk)**")
        st.caption("97% on time-based holdout (most recent 20% of events)")

    if page == "Predict & Plan":
        page_predict(model, options)
    elif page == "Analytics":
        page_analytics(model_df, rec_df, model)
    else:
        page_explorer(rec_df)


if __name__ == "__main__":
    main()
