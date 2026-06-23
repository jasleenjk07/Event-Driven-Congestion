# Gridlock Hackathon 2.0 — Round 2 Submission

**Project:** Event-Driven Congestion Risk Forecasting & Traffic Management System  
**Hackathon:** [Gridlock Hackathon 2.0 — Round 2](https://www.hackerearth.com/community/challenges/hackathon/gridlock-hackathon-20-round-2/) (Flipkart × Bengaluru Traffic Police)

---

## Problem

Bengaluru traffic police (ASTraM) handle thousands of disruptive events — protests, accidents, VIP movements, construction, tree falls — across corridors and zones. Operators must decide **how many officers to deploy, whether to barricade, and if diversions are needed**, often under time pressure and without a systematic risk score.

## Solution

An **AI-powered decision support dashboard** that:

1. Predicts **high congestion risk** from ASTraM event metadata (9 features, Random Forest)
2. Computes an interpretable **Event Congestion Index (ECI)** for operators
3. Recommends **police deployment, barricades, diversion, and monitoring**
4. Adds an independent **duration-risk model** (accident / tree_fall / congestion) from real closure timestamps

## Who uses it

**ASTraM traffic control operators** — when a new event is logged, they enter details and receive an actionable deployment plan in seconds.

## Demo flow (3–5 minutes)

1. **Predict & Plan** → Load *"Protest on ORR East — evening peak"* → HIGH risk → 20 officers, diversion required → Download CSV plan
2. **Analytics** → Show **time-based holdout** metrics (97% recall on unseen recent events)
3. **Event Explorer** → Filter high-risk events → Map clusters across Bengaluru

## Models

| Model | Purpose | Holdout performance |
|-------|---------|---------------------|
| Primary RF | High congestion risk | 74% precision, **97% recall** (time-based 20% holdout) |
| Duration RF | Long clearance time | 64% accuracy (249 events, 3 causes) |

## Data

- **8,057** historical ASTraM events (anonymized)
- Features: event type/cause, priority, road closure, hour, month, weekend, corridor, zone

## Impact

- Catches **97% of high-risk events** on the most recent 20% of data (simulates deploying on future events)
- Converts predictions into **concrete resource plans** (not just a score)
- Built on **real Bengaluru corridors** (ORR, Bellary Road, Hosur Road, etc.)

## Scalability & roadmap

| Phase | Enhancement |
|-------|-------------|
| Now | Streamlit dashboard + CSV export |
| Next | ASTraM API integration for live event ingestion |
| Future | MapmyIndia routing for diversion plans; live traffic feeds |

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd dashboard && streamlit run app.py
```

## Live demo

<!-- Add your Streamlit Cloud URL after deploy -->
`[Deploy to Streamlit Cloud and paste URL here]`

## Demo video

<!-- Add your YouTube / Drive link -->
`[Record 3–5 min screen capture and paste link here]`

## Team

<!-- Your name / team -->
`[Your name]`

## Repository structure

```
data/processed/     # Clean datasets
models/             # Trained .pkl pipelines
dashboard/          # Streamlit app
notebooks/          # EDA → features → models → recommendations
reports/figures/    # Exported charts (run scripts/generate_reports.py)
```
