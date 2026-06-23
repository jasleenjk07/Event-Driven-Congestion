# 🚦 Event-Driven Congestion Risk Forecasting and Traffic Management System

## 📌 Overview

Urban traffic disruptions caused by construction activities, public events, processions, protests, accidents, waterlogging, and VIP movements can significantly impact road networks. Traffic authorities often rely on manual assessments and historical experience to deploy resources and manage congestion.

This project presents an **Event-Driven Congestion Risk Forecasting and Traffic Management System** that predicts whether an event is likely to create high congestion risk and recommends appropriate traffic management measures such as police deployment, barricading requirements, diversion plans, and monitoring strategies.

Since direct traffic measurements (traffic speed, traffic volume, occupancy, and travel delay) were unavailable in the dataset, an **Event Congestion Index (ECI)** was engineered using event characteristics, road closure information, temporal factors, and event severity indicators. A second, independent model predicts **long-duration clearance risk** from actual event-closure timestamps.

**Live demo:** [https://event-driven-congestion.streamlit.app/](https://event-driven-congestion.streamlit.app/)

---

## 🎯 Problem Statement

**How can historical event data be used to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?**

---

## 🚀 Key Features

- Event Congestion Index (ECI) generation
- Congestion risk forecasting using Machine Learning
- Outcome-based duration risk forecasting (second model)
- High-risk event detection
- Traffic resource recommendation engine
- Police deployment planning
- Barricade requirement estimation
- Diversion strategy recommendation
- Interactive Streamlit dashboard with prediction, analytics, and event explorer views

---

## 📂 Project Structure

```text
Event-Driven-Congestion/

├── data/
│   ├── raw/
│   └── processed/
│       ├── events_cleaned.csv
│       ├── model_dataset.csv
│       ├── recommendation_dataset.csv
│       └── duration_dataset.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Recommendation_Engine.ipynb
│   ├── 05_Duration_Risk_Dataset.ipynb
│   └── 06_Duration_Model_Training.ipynb
│
├── models/
│   ├── high_congestion_risk_model.pkl
│   └── duration_risk_model.pkl
│
├── dashboard/
│   ├── app.py              # Streamlit UI
│   └── utils.py            # ECI, predictions, recommendations
│
├── reports/
│   └── figures/            # Run: python scripts/generate_reports.py
│
├── scripts/
│   └── generate_reports.py
│
├── SUBMISSION.md           # Hackathon submission brief
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Dataset Description

The dataset contains **8,057** historical traffic event records with attributes such as:

| Feature | Description |
|----------|-------------|
| event_type | Planned or unplanned event |
| event_cause | Cause of disruption |
| priority | Event priority level |
| requires_road_closure | Whether road closure is required |
| corridor | Road corridor information |
| zone | Traffic zone |
| police_station | Responsible police station |
| latitude / longitude | Event location |
| start_datetime | Event start timestamp |
| closed_datetime | Event closure timestamp (when available) |

### Major Event Categories

- Vehicle Breakdown
- Construction
- Water Logging
- Accident
- Tree Fall
- Public Event
- Procession
- VIP Movement
- Protest
- Others

---

## 🛠 Feature Engineering

### Temporal Features

- Hour of Day
- Day of Week
- Month
- Weekend Indicator

### Event Severity Features

- Priority Score
- Cause Score
- Closure Score
- Time Score
- Weekend Score

Peak-hour scoring applies during **07:00–10:00** and **17:00–20:00**.

---

## 📈 Event Congestion Index (ECI)

An Event Congestion Index was created to estimate expected congestion severity.

```text
ECI =
  Priority Score
+ Cause Score
+ Closure Score
+ Time Score
+ Weekend Score
```

### Congestion Risk Levels

| ECI Range | Risk Level |
|-----------|------------|
| ECI < 6 | Low |
| 6 ≤ ECI < 10 | Medium |
| ECI ≥ 10 | High |

The ML target variable `high_congestion_risk` is derived from ECI: **1** when ECI risk level is **High**, otherwise **0**.

---

## 🤖 Primary Model: Congestion Risk

### Algorithm

Random Forest Classifier (300 estimators, max depth 10, balanced class weights)

Preprocessing pipeline:

- One-hot encoding for categorical features
- Passthrough for numeric features

### Input Features (9)

- Event Type
- Event Cause
- Priority
- Requires Road Closure
- Hour
- Month
- Weekend Indicator
- Corridor
- Zone

### Target Variable

```text
high_congestion_risk
```

Where:

```text
1 = High Congestion Risk
0 = Normal Congestion Risk
```

### Train / Test Split

Time-based holdout: events sorted by `start_datetime`, with the **most recent 20%** held out for testing (1,612 test events).

---

## 📊 Primary Model Performance

### Test Set (time-based holdout)

```text
              precision    recall  f1-score   support

           0       1.00      0.99      0.99      1552
           1       0.74      0.97      0.84        60

    accuracy                           0.99      1612
   macro avg       0.87      0.98      0.92      1612
weighted avg       0.99      0.99      0.99      1612
```

### Confusion Matrix (test set)

```text
[[1532   20]
 [   2   58]]
```

### Full Dataset Validation (dashboard — in-sample, reference only)

| Metric | Value |
|--------|-------|
| Accuracy | 99.2% |
| High-risk precision | 79.2% |
| High-risk recall | 99.2% |
| False positives | 65 |
| False negatives | 2 |

> The dashboard **Analytics** page leads with **time-based holdout** metrics (recommended for judging). Full-dataset numbers are in-sample and shown for reference only.

### Key Observation

After adding `requires_road_closure` as a model feature and switching to a time-based train/test split, the model catches **97% of high-congestion-risk events** on the held-out test set with far fewer false alarms than the original random-split version (precision rose from 0.31 to 0.74).

---

## 🧪 Second Model: Outcome-Based Duration Risk

The primary congestion model learns to predict a label derived from the hand-built ECI rule (a plain lookup table achieves 100% accuracy on that label with no ML). To provide a genuinely independent signal, a second model was trained on **actual event-closure durations**:

```text
duration_min = closed_datetime - start_datetime
```

### Scope

Only causes where `closed_datetime` reliably reflects real road-clearance time:

| Cause | Events in dataset |
|-------|-------------------|
| tree_fall | 140 |
| accident | 87 |
| congestion | 22 |

**Excluded causes:** vehicle_breakdown, protest, and procession (sample-size and distributional concerns); construction, potholes, debris, and water logging (closure timestamps reflect administrative ticket closure, not physical road clearance).

### Dataset

- **249 closed events** with valid start and closure timestamps
- Duration capped at 7 days; negative durations removed
- **~25% positive rate** for `high_duration_risk`

### Target

`high_duration_risk` = **1** if an event's closure duration exceeded the **75th percentile** for its cause category, otherwise **0**.

### Input Features

Same 9 features as the primary model.

### Train / Test Split

Random stratified split (80/20) → **199 train / 50 test** events.

### Test Set Performance

```text
              precision    recall  f1-score   support

           0       0.74      0.78      0.76        37
           1       0.27      0.23      0.25        13

    accuracy                           0.64        50
   macro avg       0.51      0.51      0.51        50
weighted avg       0.62      0.64      0.63        50
```

### Confusion Matrix (test set)

```text
[[29  8]
 [10  3]]
```

The duration model is shown in the dashboard for **accident**, **tree_fall**, and **congestion** only.

---

## ⚠️ Known Limitations

- **ECI is rule-based**, not learned from traffic measurements. The primary ML model largely reconstructs this deterministic label, especially after `requires_road_closure` was added as a feature (it is the strongest predictor).
- **No live traffic data** (speed, volume, occupancy) is available; all signals come from event metadata.
- **Duration model accuracy is moderate** (~64% on test set, 50 events) and limited to accident, tree_fall, and congestion
- **Closure timestamps** for some event types reflect ticket workflow, not physical road clearance.

---

## 🚔 Recommendation Engine

Based on predicted congestion risk and event characteristics, the system recommends officers, barricades, diversion plans, and monitoring status. Logic is implemented in `dashboard/utils.py` and matches notebook 04.

### Police Deployment (high-risk events)

| Event Cause | Officers |
|-------------|----------|
| VIP Movement | 20 (+2 if High priority) |
| Protest | 18 (+2 if High priority) |
| Procession | 15 (+2 if High priority) |
| Public Event | 12 (+2 if High priority) |
| Construction | 8 (+2 if High priority) |
| Accident | 8 (+2 if High priority) |
| Water Logging | 6 (+2 if High priority) |

For **normal-risk** predictions: **2 officers** (Low priority) or **4 officers** (High priority).

### Barricade Requirements (high-risk events)

| Event Cause | Barricades |
|-------------|------------|
| Construction | 12 |
| Water Logging | 10 |
| VIP Movement | 10 |
| Protest | 10 |
| Procession | 8 |
| Public Event | 8 |
| Tree Fall | 6 |
| Accident | 4 |
| Vehicle Breakdown | 2 |

### Diversion Planning

| Outcome | When |
|---------|------|
| Required | VIP, protest, procession, public event, construction, water logging |
| Optional | Default for other high-risk predictions |
| Not Required | Normal-risk predictions |

---

## 💻 Streamlit Dashboard

Interactive dashboard for traffic operators with three pages:

| Page | Purpose |
|------|---------|
| **Predict & Plan** | Enter event details, run ML prediction, view ECI breakdown, duration risk (where applicable), and resource recommendations |
| **Analytics** | Historical charts and model validation metrics |
| **Event Explorer** | Filter historical events; marker + heatmap views across Bengaluru |

### Predict & Plan capabilities

1. **Demo scenarios** — pre-loaded Bengaluru cases (protest/ORR, VIP/Bellary Road, tree fall)
2. Enter event details (type, cause, priority, corridor, zone, datetime, road closure)
3. Predict congestion risk with probability score
4. View ECI component breakdown
5. Receive deployment recommendations (officers, barricades, diversion, monitoring)
6. **Download deployment plan (CSV)**
7. View outcome-based duration risk for accident, tree_fall, and congestion
8. **Auto-map** after prediction — pin, deployment popup, and influence zone on Bengaluru map
9. Alignment warnings when ML prediction and ECI level disagree

### Maps (Bengaluru)

| Page | Map features |
|------|----------------|
| **Predict & Plan** | Event pin with corridor/zone popup and 800 m influence circle |
| **Analytics** | High-risk heatmap + marker/heatmap toggle |
| **Event Explorer** | Markers, heatmap, or both; click for ECI and deployment details |

### Sample Output

```text
Congestion Risk: HIGH

Risk Probability: 82%

ECI: 13 (High)

Duration Risk: LONG DURATION RISK (68% probability)

Recommended Officers: 20

Barricades: 10

Diversion: Required

Monitoring: Yes
```

### Dashboard validation

Dashboard logic was verified against processed datasets:

- ECI calculation: **100% match** with `model_dataset.csv`
- Recommendation engine: **100% match** with `recommendation_dataset.csv`
- Live ML predictions: **100% match** with notebook outputs

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd Event-Driven-Congestion
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

```text
pandas, numpy, matplotlib, seaborn, scikit-learn,
streamlit, folium, joblib
```

Versions are pinned in `requirements.txt` for reproducibility.

---

## 🏁 Hackathon Submission (Gridlock 2.0)

**Live app:** [https://event-driven-congestion.streamlit.app/](https://event-driven-congestion.streamlit.app/)

See **[SUBMISSION.md](SUBMISSION.md)** for the judging brief, demo script, and submission details.

### Quick demo scenarios (dashboard)

| Scenario | What it shows |
|----------|----------------|
| Protest on ORR East — evening peak | High congestion risk + full deployment plan |
| VIP movement on Bellary Road | VIP resource rules + road closure |
| Tree fall with road closure | Duration-risk model (independent signal) |

### Export figures for slides

```bash
python scripts/generate_reports.py
```

Charts are saved to `reports/figures/`.

### Deployed on Streamlit Cloud

| Setting | Value |
|---------|--------|
| URL | [event-driven-congestion.streamlit.app](https://event-driven-congestion.streamlit.app/) |
| Main file | `dashboard/app.py` |
| Repository | [jasleenjk07/Event-Driven-Congestion](https://github.com/jasleenjk07/Event-Driven-Congestion) |

To redeploy after changes: push to `main` on GitHub — Streamlit Cloud rebuilds automatically.

---

## ▶️ Run Dashboard

### Live (recommended for judges)

**[https://event-driven-congestion.streamlit.app/](https://event-driven-congestion.streamlit.app/)**

### Local development

From the project root with the virtual environment activated:

```bash
cd dashboard
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

> Use the project `venv` so all dependencies (including `folium`) are available. If you run Streamlit from another Python environment, install requirements there as well.

---

## 📓 Notebooks Workflow

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Exploratory data analysis |
| `02_Feature_Engineering.ipynb` | Temporal features, ECI, target creation |
| `03_Model_Training.ipynb` | Primary Random Forest (time-based split, 9 features) |
| `04_Recommendation_Engine.ipynb` | Rule-based resource recommendations |
| `05_Duration_Risk_Dataset.ipynb` | Build duration dataset (accident, tree_fall, congestion) |
| `06_Duration_Model_Training.ipynb` | Second Random Forest for duration risk (249 events) |

Run notebooks in order after placing raw data in `data/raw/`.

---

## 🔮 Future Enhancements

- Integration with Google Maps Traffic API
- Real-time traffic speed monitoring
- Weather data integration
- Holiday and event calendar integration
- Dynamic route optimization
- Deep learning-based congestion forecasting
- Live traffic camera integration

---

## 🏆 Conclusion

This project demonstrates how historical event data can be transformed into actionable traffic intelligence. By combining feature engineering, machine learning, and rule-based decision support, the system helps traffic authorities proactively identify high-risk events and optimize deployment of traffic management resources.
# Event-Driven-Congestion
