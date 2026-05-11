# RWA Pricing Deviation Monitor

Real-time dashboard for OUSG (treasury token) and MPL (governance token).

**Live demo**:

## What it does
- Monitors gap between market price and NAV
- Alerts when |gap| > 0.5%
- Experimental health score for MPL

## Key finding
MPL has a persistent ~10% discount that volume and TVL cannot explain (R² = 0.075).

## Run locally
```bash
pip install -r requirements.txt
streamlit run rwa_dashboard.py
