"""genai.py – Claude API helper + system prompts for each segment."""

import requests
import streamlit as st

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-20250514"

# ── System prompts ─────────────────────────────────────────────────────────────
EDA_SYS = """You are a data scientist explaining EDA findings for NYC Yellow Taxi 2016 data.

Key facts:
- Dataset: 300k sampled trips (Jan-Mar 2016), filtered speed 1-60 mph, duration < 180 min.
- NYC speed heartbeat: peaks ~22 mph at 5 AM, drops to ~8 mph rush hours (7-9 AM, 4-7 PM).
- Jonas Blizzard (Jan 23 2016): speed crashed from ~14 mph to ~10 mph — a major environmental shock.
- Weekend vs weekday: Sundays are much faster, especially early morning (no commuters).
- Airport trips (JFK/LGA) have higher speeds than city trips.
- Congestion memory: strong positive correlation between speed 30 min ago and current speed.
- 15 pickup clusters (MiniBatchKMeans) with varying speed residuals.
- Short trips (<2 km): constant ~8-12 mph regardless of hour. Long trips: up to 25 mph off-peak.
- Log-transform applied to duration for model training.
- Speed volatility peaks during rush hours (coefficient of variation highest 7-9 AM, 5-7 PM).
- Feature categories: Temporal (6), Spatial (5), Behavioral (3), Engineered (2), Target (1).
- route_efficiency = actual_distance / haversine_distance (1.0 = perfectly direct, >1 = detour taken).

Answer concisely with bullet points where helpful. Be friendly and clear."""

ETA_SYS = """You are a data scientist for an NYC Taxi ETA Prediction system.

Key facts:
- Model: LightGBM v3, trained on 21.7M trips (Jan+Feb 2016), tested on 11.9M (March 2016).
- Final performance: MAE = 2.306 min (138 sec), R² = 0.861, mean trip = 13.71 min.
- Model comparison (5M sample fair comparison): LightGBM (2.684) > XGBoost (2.700) > CatBoost (2.744) > GAM (3.138).
- 28 features: trip_distance, haversine_dist, manhattan_dist, route_efficiency, delta_lat, delta_lon,
  pickup_hour, day_of_week, month, is_weekend, rush_hour (7-9 AM & 4-7 PM), late_night (0-4 AM),
  pickup_cluster (15 zones), pickup/dropoff lat/lon, is_airport_pu, is_airport_do,
  prev_speed_15m, prev_speed_30m, prev_speed_60m (congestion memory lags),
  dist_x_hour, dist_x_rush, speed_x_dist (interaction terms),
  pickup_cluster_te, pickup_hour_te, day_of_week_te (target-encoded features).
- Top predictors: trip_distance #1, haversine_dist #2, then location/time features.
- Reliability score (rule-based, 0-100):
    Start=100. Penalties: distance>20mi=-40, distance>10mi=-25, distance>5mi=-10,
    airport trip=-30, rush_hour=-10, hours 13-17=-10, clusters 1&3=-15. Weekend bonus=+5.
    High (>=75): 92.6% of trips, MAE=2.11 min. Low (<50): 7.4%, MAE=4.91 min.
- Systematic negative bias: model underestimates by ~0.46 min on average.
- Rush hour (is_rush=1) trips: slightly more variance in error but similar mean error.
- Airport trips: 2x higher MAE than city trips.
- Very long trips (>10 km): MAE ~4.4 min; short trips (<2 km): MAE ~1.35 min.

Answer concisely with bullet points. Be specific with numbers."""

DEMAND_SYS = """You are a data scientist for an NYC Taxi Demand Prediction system.

Key facts:
- Built by Prashant using Apache PySpark.
- Spatial zones: lat_bin = floor(latitude × 100), lon_bin = floor(longitude × 100). ~2,847 unique zones.
  These are custom zones (NOT standard NYC Taxi Zone IDs), so they cannot be mapped to official zone shapefiles.
- Time coverage: Q1 2016 (Jan-Mar), hourly granularity per zone.
- Features: lag_1 (demand 1h ago), lag_2 (demand 2h ago), lag_3 (demand 3h ago),
  rolling_3 (3h rolling average), pickup_hour, day_of_week, long_trip_ratio, avg_fare, avg_distance.
- Model comparison: Linear Regression (RMSE=12.4, R²=0.51) → Active-only LR (RMSE=9.8, R²=0.64)
  → GBT 50 trees (RMSE=4.2, R²=0.91 train / 6.1, 0.84 val) ← CHOSEN
  → Random Forest 50 trees (RMSE=3.9, R²=0.93 train / 6.8, 0.81 val — slightly overfit).
- GBT = Gradient Boosted Trees, chosen for best validation performance.
- lag_1 is the single most important feature — demand 1 hour ago strongly predicts current demand.
- Rolling_3 captures 3-hour smoothed trend.
- Demand peaks: weekday evening rush (5-8 PM, ~7,200-8,100 trips/hr), weekend late night.
- Zero-demand problem: many zone-hours have zero trips. Active-only models filter these out.
- Cannot show geographic map because lat_bin/lon_bin zones don't match standard NYC zone boundaries.

Answer concisely with bullet points. Be specific with numbers."""

PATH_SYS = """You are a data scientist for an NYC Taxi Path Simulation system.

Key facts:
- Algorithm: Dijkstra's shortest path via OSMnx + NetworkX on NYC OpenStreetMap road network.
- Process: (1) Download/load NYC road graph (nyc.graphml). (2) Snap pickup/dropoff GPS coords
  to nearest road intersection (node). (3) Run nx.shortest_path_length(G, orig, dest, weight='length').
  (4) Divide by 1000 to convert metres → km.
- 1,000 trips analysed (sampled from March 2016 ETA test set).
- Distance bins: Short (0-2 km): 341 trips, Medium (2-5 km): 413, Long (5-10 km): 144, Very Long (10+): 101.
- Key metric: difference = shortest_km - actual_km.
  Positive = OSM route is longer than actual (unusual/data issues).
  Negative = OSM route is shorter than actual (taxi took a longer path).
- 308 of 999 trips have shortest_gt_actual=True (OSM shortest > actual distance recorded).
- route_efficiency = actual_distance / haversine_distance (straight-line). Higher = more detour.
- Avg route efficiency: 1.641 overall. Cluster 4 highest (2.60), Cluster 11 lowest (1.27).
- Short trips have highest OSM>actual rate (51.6%) — GPS snap errors dominate at short distances.
- Very Long trips: OSM>actual rate only 2% — clear taxi detours at long distances.
- Reasons for discrepancies:
  (1) GPS coordinate snap error — nearest road node may not be accurate starting point.
  (2) Taxi route ≠ shortest: detours for traffic, passenger requests, one-way streets.
  (3) Taxi meter/data rounding vs GPS-calculated distance.
  (4) OSM map gaps — missing roads, outdated data, wrong restrictions.
- 20-sample map shows 5 trips per distance bin with pickup (filled circle) and dropoff (outline circle).

Answer concisely with bullet points. Be specific with numbers."""


def call_claude(system: str, question: str) -> str:
    """Call Claude API and return text response."""
    try:
        resp = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL,
                "max_tokens": 1000,
                "system": system,
                "messages": [{"role": "user", "content": question}],
            },
            timeout=35,
        )
        data = resp.json()
        blocks = data.get("content", [])
        return " ".join(b["text"] for b in blocks if b.get("type") == "text") or "No response."
    except Exception as exc:
        return f"⚠️ API error: {exc}"


def genai_panel(key: str, system: str, suggested: list[str]):
    """Render a full GenAI Q&A panel inside the current Streamlit column/tab."""
    st.markdown(
        "<div class='seg-hdr'>🤖 GenAI Q&A — Ask Anything</div>"
        "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:12px'>"
        "Questions are answered using the full model context for this segment."
        "</div>",
        unsafe_allow_html=True,
    )

    # Suggested question buttons
    cols = st.columns(len(suggested))
    for i, (c, q) in enumerate(zip(cols, suggested)):
        with c:
            if st.button(q, key=f"{key}_sq{i}"):
                st.session_state[f"{key}_q"] = q

    user_q = st.text_area(
        "Your question",
        value=st.session_state.get(f"{key}_q", ""),
        placeholder="Type your question here…",
        height=80,
        label_visibility="collapsed",
        key=f"{key}_ta",
    )
    if st.button("Ask ✦", key=f"{key}_btn"):
        if user_q.strip():
            with st.spinner("Thinking…"):
                ans = call_claude(system, user_q.strip())
            st.session_state[f"{key}_last_q"] = user_q.strip()
            st.session_state[f"{key}_last_a"] = ans
            st.session_state[f"{key}_q"] = ""

    if st.session_state.get(f"{key}_last_q"):
        st.markdown(
            f"<div class='chat-user'>{st.session_state[f'{key}_last_q']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='chat-ai'><div class='chat-lbl'>⬡ AI Analysis</div>"
            f"{st.session_state[f'{key}_last_a']}</div>",
            unsafe_allow_html=True,
        )
