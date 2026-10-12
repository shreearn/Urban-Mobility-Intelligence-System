"""seg_demand.py – Segment 3: Demand Prediction"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import time

from styles import DARK, YELLOW, BLUE, GREEN, RED, ORANGE, MUTED, seg_header, kpi, kpi_animated, sim_card
from genai import genai_panel, DEMAND_SYS


def render(demand_df: pd.DataFrame, is_full: bool):
    st.markdown("<div class='pg-title'>Demand Prediction</div>", unsafe_allow_html=True)
    src = f"Full Parquet ({len(demand_df):,} rows)" if is_full else "Representative Analysis (parquet not loaded)"
    st.markdown(
        f"<div class='pg-sub'>PySpark GBT · lat_bin / lon_bin Zones · Q1 2016 · {src}</div>",
        unsafe_allow_html=True,
    )
    if not is_full:
        st.info(
            "📂 **Demand parquet not found.** Charts are built from notebook-derived representative data. "
            "Place `final_results_demand_pred_2016.parquet` in the `data/` folder to unlock live data.",
            icon="ℹ️",
        )

    tab_viz, tab_time, tab_forecast, tab_model, tab_genai = st.tabs(
        ["Visualisations", "Hourly Explorer", "Forecast Playground",
         "Model Comparison", "GenAI Q&A"]
    )

    hours = list(range(24))

    # ── If we have real parquet ───────────────────────────────────────────────
    if is_full and len(demand_df) > 0:
        # Real data path
        if "timestamp" in demand_df.columns:
            demand_df["pickup_hour"] = pd.to_datetime(demand_df["timestamp"]).dt.hour
            demand_df["day_of_week"] = pd.to_datetime(demand_df["timestamp"]).dt.dayofweek
        if "demand" in demand_df.columns:
            hourly_real = demand_df.groupby("pickup_hour")["demand"].mean().reindex(range(24))
        else:
            hourly_real = None
    else:
        hourly_real = None

    # ── Synthetic representative data (from notebook findings) ────────────────
    demand_wd = [3800,2100,1400,1100,900,1200,2800,5200,6400,5800,5200,5600,
                 6000,5800,5500,5200,5800,7200,8100,7800,7200,6800,6200,5200]
    demand_we = [5200,5800,6200,6100,5200,4100,3200,2800,3100,4200,5400,6200,
                 6800,7000,6900,6800,6900,7400,8200,8000,7800,7500,7000,6200]

    with tab_viz:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        for col, (v, l) in zip(
            [c1, c2, c3, c4],
            [
                ("34.5M", "Total 2016 Trips"),
                ("2847", "Unique Zones"),
                ("GBT", "Best Model"),
                ("lag_1", "#1 Feature"),
            ],
        ):
            col.markdown(kpi_animated(v, l), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Hourly demand + Peak vs non-peak ───────────────────────────
        cl, cr = st.columns(2)
        with cl:
            st.markdown(seg_header("Hourly Demand — Weekday vs Weekend"), unsafe_allow_html=True)
            fig = go.Figure()
            y_wd = hourly_real.values if hourly_real is not None else demand_wd
            fig.add_trace(go.Scatter(x=hours, y=y_wd, name="Weekday", mode="lines",
                                     line=dict(color=BLUE, width=2.5),
                                     fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
            fig.add_trace(go.Scatter(x=hours, y=demand_we, name="Weekend", mode="lines",
                                     line=dict(color=YELLOW, width=2.5),
                                     fill="tozeroy", fillcolor="rgba(245,197,24,0.08)"))
            fig.add_vrect(x0=7, x1=9,   fillcolor=RED, opacity=0.07, line_width=0)
            fig.add_vrect(x0=17, x1=20, fillcolor=RED, opacity=0.07, line_width=0)
            fig.update_layout(**DARK, height=300,
                              xaxis_title="Hour of Day", yaxis_title="Avg Trip Count")
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown(seg_header("Peak vs Non-Peak Demand Distribution"), unsafe_allow_html=True)
            np.random.seed(5)
            peak_d    = np.random.normal(7200, 800, 300)
            nonpeak_d = np.random.normal(3800, 600, 300)
            fig = go.Figure()
            fig.add_trace(go.Violin(y=peak_d, name="Peak Hours (7-9,17-20)",
                                    side="negative", line_color=RED,
                                    fillcolor="rgba(239,68,68,0.20)", meanline_visible=True))
            fig.add_trace(go.Violin(y=nonpeak_d, name="Non-Peak",
                                    side="positive", line_color=GREEN,
                                    fillcolor="rgba(34,197,94,0.20)", meanline_visible=True))
            fig.update_layout(**DARK, height=300, violingap=0, violinmode="overlay",
                              yaxis_title="Demand (trips/hr)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Feature importance + Lag correlation ───────────────────────
        cl2, cr2 = st.columns(2)
        with cl2:
            st.markdown(seg_header("GBT Feature Importance"), unsafe_allow_html=True)
            feats = ["lag_1","rolling_3","lag_2","lag_3","pickup_hour",
                     "day_of_week","long_trip_ratio","avg_fare","avg_distance","lat_bin","lon_bin"]
            imps  = [0.285,0.198,0.162,0.098,0.072,0.058,0.042,0.031,0.025,0.019,0.010]
            fig = go.Figure(go.Bar(
                x=imps[::-1], y=feats[::-1], orientation="h",
                marker=dict(color=imps[::-1], colorscale=[[0, BLUE],[1, YELLOW]]),
                text=[f"{v:.3f}" for v in imps[::-1]], textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.update_layout(**DARK, height=340, showlegend=False,
                              xaxis_title="Importance Score", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with cr2:
            st.markdown(seg_header("Demand Lag Correlation (lag_1 vs current)"), unsafe_allow_html=True)
            np.random.seed(7)
            lag1 = np.random.uniform(0, 100, 500)
            curr = lag1 * 0.85 + np.random.normal(0, 8, 500)
            curr = np.clip(curr, 0, 120)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lag1, y=curr, mode="markers",
                                     marker=dict(color=BLUE, size=4, opacity=0.5)))
            m, b = np.polyfit(lag1, curr, 1)
            xr = np.array([0, 100])
            fig.add_trace(go.Scatter(x=xr, y=m * xr + b, mode="lines",
                                     line=dict(color=YELLOW, width=2), name="Trend"))
            fig.update_layout(**DARK, height=340, showlegend=False,
                              xaxis_title="Demand 1h Ago (lag_1)",
                              yaxis_title="Current Demand")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: Rolling average ─────────────────────────────────────────────
        st.markdown(seg_header("Rolling Average Demand Smoothing — Sample Zone (March 2016)"), unsafe_allow_html=True)
        np.random.seed(99)
        ts = pd.date_range("2016-03-01", "2016-03-31 23:00", freq="h")
        raw_d = np.random.poisson(lam=45, size=len(ts)).astype(float)
        raw_d[120:132] += 70  # demand spike
        roll3 = pd.Series(raw_d).rolling(3).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=raw_d, mode="lines", name="Raw Demand",
                                  line=dict(color=BLUE, width=1), opacity=0.55))
        fig.add_trace(go.Scatter(x=ts, y=roll3, mode="lines", name="Rolling-3h Avg",
                                  line=dict(color=YELLOW, width=2)))
        fig.update_layout(**DARK, height=260, xaxis_title="Time",
                          yaxis_title="Demand (trips/hr)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Row 4: Day-of-week pattern ────────────────────────────────────────
        st.markdown(seg_header("Daily Demand Pattern Across Week"), unsafe_allow_html=True)
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        avg_demand_by_day = [5100, 5300, 5500, 5600, 6200, 7100, 6500]
        fig = go.Figure(go.Bar(
            x=days, y=avg_demand_by_day,
            marker_color=[RED if d in ["Fri","Sat"] else BLUE for d in days],
            text=avg_demand_by_day, textposition="outside",
            textfont=dict(color="#e8ecf4"),
        ))
        fig.update_layout(**DARK, height=260, showlegend=False,
                          xaxis_title="Day of Week", yaxis_title="Avg Daily Trips")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Hourly Explorer ────────────────────────────────────────────────
    with tab_time:
        st.markdown(seg_header("Animated Hour-by-Hour Demand Explorer"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Explore how demand evolves through the day. Use the slider or auto-play.</div>",
            unsafe_allow_html=True,
        )

        dc1, dc2 = st.columns([4, 1])
        with dc1:
            dem_hour = st.slider("Select Hour", 0, 23, 12, key="dem_hour_slider")
        with dc2:
            dem_auto = st.button("▶ Auto-Play", key="dem_autoplay")

        h = dem_hour

        # Time period label
        period = "Late Night" if h < 5 else "Early Morning" if h < 7 else \
                 "AM Rush" if h < 10 else "Midday" if h < 16 else \
                 "PM Rush" if h < 19 else "Evening" if h < 22 else "Late Night"

        is_peak = (7 <= h <= 9) or (17 <= h <= 20)

        # KPI row
        dk1, dk2, dk3, dk4 = st.columns(4)
        wd_color = RED if is_peak else GREEN
        dk1.markdown(sim_card(f"{demand_wd[h]:,}", "Weekday Demand", wd_color), unsafe_allow_html=True)
        dk2.markdown(sim_card(f"{demand_we[h]:,}", "Weekend Demand", YELLOW), unsafe_allow_html=True)
        diff = demand_we[h] - demand_wd[h]
        diff_color = GREEN if diff > 0 else RED
        dk3.markdown(sim_card(f"{diff:+,}", "Weekend Δ", diff_color), unsafe_allow_html=True)
        dk4.markdown(sim_card(period, f"Hour {h}:00", MUTED,
                              "Peak" if is_peak else "Off-Peak"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        tl, tr = st.columns(2)
        with tl:
            st.markdown(seg_header(f"Demand Comparison at {h}:00"), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Weekday", "Weekend"],
                                 y=[demand_wd[h], demand_we[h]],
                                 marker_color=[BLUE, YELLOW],
                                 text=[f"{demand_wd[h]:,}", f"{demand_we[h]:,}"],
                                 textposition="outside", textfont=dict(color="#e8ecf4")))
            fig.update_layout(**DARK, height=300, showlegend=False,
                              yaxis_title="Trips/Hour")
            st.plotly_chart(fig, use_container_width=True)

        with tr:
            st.markdown(seg_header(f"Demand Context: All Hours (current = {h}:00)"), unsafe_allow_html=True)
            colors = [YELLOW if i == h else (RED if (7 <= i <= 9 or 17 <= i <= 20) else BLUE)
                      for i in range(24)]
            fig = go.Figure(go.Bar(x=list(range(24)), y=demand_wd,
                                   marker_color=colors))
            fig.add_vline(x=h, line_color=YELLOW, line_width=3, opacity=0.8)
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Hour of Day", yaxis_title="Weekday Trips/Hour")
            st.plotly_chart(fig, use_container_width=True)

        # Auto-play
        if dem_auto:
            placeholder = st.empty()
            for auto_h in range(24):
                with placeholder.container():
                    st.markdown(
                        f"<div style='text-align:center;color:#f5c518;font-family:Space Mono;font-size:1.2rem'>"
                        f"▶ {auto_h:02d}:00 — Weekday: {demand_wd[auto_h]:,} | "
                        f"Weekend: {demand_we[auto_h]:,} | "
                        f"Δ: {demand_we[auto_h]-demand_wd[auto_h]:+,}</div>",
                        unsafe_allow_html=True,
                    )
                time.sleep(0.4)
            placeholder.markdown(
                "<div style='text-align:center;color:#22c55e;font-size:.9rem'>✓ Playback complete</div>",
                unsafe_allow_html=True,
            )

    # ── Tab 3: Forecast Playground ────────────────────────────────────────────
    with tab_forecast:
        st.markdown(seg_header("Demand Forecast Playground"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Adjust zone and time parameters to see predicted demand. "
            "Uses pre-aggregated statistics from the demand model.</div>",
            unsafe_allow_html=True,
        )

        from data_loader import load_demand_lookup
        dem_lookup = load_demand_lookup()

        if len(dem_lookup) == 0:
            st.warning("⚠️ Demand parquet not available. Showing representative estimates.")
            # Fallback using synthetic data
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                fg_day = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                      index=0, key="dem_fg_day")
            with fc2:
                fg_hour = st.slider("Hour", 0, 23, 12, key="dem_fg_hour")
            with fc3:
                fg_lag1 = st.slider("Lag-1 Demand (1h ago)", 0, 200, 50, key="dem_fg_lag1")

            day_idx = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(fg_day)
            is_we = day_idx >= 5
            base = demand_we[fg_hour] if is_we else demand_wd[fg_hour]
            # Simple synthetic prediction influenced by lag
            pred = base * 0.6 + fg_lag1 * 0.4 * 40
            pred = max(0, pred)

            rk1, rk2, rk3 = st.columns(3)
            rk1.markdown(sim_card(f"{pred:,.0f}", "Predicted Demand", YELLOW,
                                  f"{fg_day} at {fg_hour}:00"), unsafe_allow_html=True)
            rk2.markdown(sim_card(f"{base:,}", "Historical Avg", BLUE), unsafe_allow_html=True)
            rk3.markdown(sim_card(f"{fg_lag1}", "Lag-1 Input", MUTED), unsafe_allow_html=True)
        else:
            # Real data path
            fc1, fc2 = st.columns(2)
            with fc1:
                fg_zone = st.selectbox("Zone Type",
                                       sorted(dem_lookup["zone_type"].dropna().unique()),
                                       key="dem_fg_zone")
                fg_day = st.selectbox("Day of Week",
                                      ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                      index=0, key="dem_fg_day_r")
            with fc2:
                fg_hour = st.slider("Hour", 0, 23, 12, key="dem_fg_hour_r")
                fg_lag1 = st.slider("Lag-1 Demand (1h ago)", 0, 200, 50, key="dem_fg_lag1_r")

            day_idx = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(fg_day)
            is_weekend = 1 if day_idx >= 5 else 0
            is_peak = 1 if (7 <= fg_hour <= 9 or 17 <= fg_hour <= 20) else 0

            match = dem_lookup[
                (dem_lookup["zone_type"] == fg_zone) &
                (dem_lookup["day_of_week"] == day_idx) &
                (dem_lookup["hour"] == fg_hour)
            ]

            if len(match) > 0:
                avg_pred = match["avg_prediction"].mean()
                avg_demand = match["avg_demand"].mean()
                std_demand = match["std_demand"].mean()
                avg_lag = match["avg_lag1"].mean()
                count = int(match["count"].sum())

                # Adjust by lag-1 influence (lag_1 is the strongest feature)
                lag_factor = fg_lag1 / max(avg_lag, 1) if avg_lag > 0 else 1
                adjusted_pred = avg_pred * (0.6 + 0.4 * min(lag_factor, 3))
            else:
                adjusted_pred = fg_lag1 * 0.85
                avg_demand = adjusted_pred
                std_demand = 10
                count = 0

            st.markdown("<br>", unsafe_allow_html=True)

            rk1, rk2, rk3, rk4 = st.columns(4)
            color_d = RED if adjusted_pred > 80 else YELLOW if adjusted_pred > 30 else GREEN
            rk1.markdown(sim_card(f"{adjusted_pred:,.1f}", "Predicted Demand", color_d,
                                  f"{fg_day} at {fg_hour}:00"), unsafe_allow_html=True)
            rk2.markdown(sim_card(f"{avg_demand:,.1f}", "Avg Historical", BLUE), unsafe_allow_html=True)
            rk3.markdown(sim_card(f"±{std_demand:,.1f}", "Std Deviation", MUTED), unsafe_allow_html=True)
            rk4.markdown(sim_card(f"{count:,}", "Data Points", MUTED,
                                  f"Zone: {fg_zone}"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence band chart
            st.markdown(seg_header("24-Hour Demand Profile"), unsafe_allow_html=True)
            hour_profile = dem_lookup[
                (dem_lookup["zone_type"] == fg_zone) &
                (dem_lookup["day_of_week"] == day_idx)
            ].groupby("hour").agg(
                avg=("avg_prediction", "mean"),
                std=("std_demand", "mean"),
            ).reindex(range(24)).fillna(0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=(hour_profile["avg"] + hour_profile["std"]).values,
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=(hour_profile["avg"] - hour_profile["std"]).clip(0).values,
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(59,130,246,0.15)", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=hour_profile["avg"].values,
                mode="lines+markers", name="Avg Prediction",
                line=dict(color=YELLOW, width=2.5), marker=dict(size=4),
            ))
            fig.add_vline(x=fg_hour, line_color=RED, line_dash="dash",
                          annotation_text=f"Selected: {fg_hour}:00",
                          annotation_font_color=RED)
            fig.update_layout(**DARK, height=300, xaxis_title="Hour of Day",
                              yaxis_title="Predicted Demand")
            st.plotly_chart(fig, use_container_width=True)

            # Zone type heatmap
            st.markdown(seg_header("Zone Type × Hour Demand Heatmap"), unsafe_allow_html=True)
            zone_types = sorted(dem_lookup["zone_type"].dropna().unique())
            heat_data = []
            for zt in zone_types:
                row_data = dem_lookup[
                    (dem_lookup["zone_type"] == zt) &
                    (dem_lookup["day_of_week"] == day_idx)
                ].groupby("hour")["avg_prediction"].mean().reindex(range(24)).fillna(0).values
                heat_data.append(row_data)

            if heat_data:
                fig = go.Figure(go.Heatmap(
                    z=heat_data, x=list(range(24)), y=zone_types,
                    colorscale="YlOrRd",
                    colorbar=dict(title=dict(text="Demand", font=dict(color="#e8ecf4")),
                                  tickfont=dict(color="#e8ecf4")),
                ))
                fig.update_layout(**DARK, height=250, xaxis_title="Hour of Day",
                                  yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Model Comparison ───────────────────────────────────────────────
    with tab_model:
        st.markdown(seg_header("Model Performance Comparison (PySpark)"), unsafe_allow_html=True)
        model_df = pd.DataFrame({
            "Model":       ["Linear Regression (all data)",
                            "Linear Regression (active only)",
                            "GBT — 50 trees ✓",
                            "Random Forest — 50 trees"],
            "Train RMSE":  [12.4, 9.8, 4.2, 3.9],
            "Val RMSE":    [12.8, 10.1, 6.1, 6.8],
            "Train R²":    [0.51, 0.64, 0.91, 0.93],
            "Val R²":      [0.48, 0.61, 0.84, 0.81],
            "Notes":       ["Baseline — too many zero-demand rows",
                            "Better — filters inactive zone-hours",
                            "CHOSEN — best validation performance",
                            "Good but slightly overfits training"],
        })
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        cl, cr = st.columns(2)
        with cl:
            st.markdown(seg_header("Validation RMSE Comparison"), unsafe_allow_html=True)
            models = ["Linear\n(all)","Linear\n(active)","GBT ✓","Random\nForest"]
            rmse_v = [12.8, 10.1, 6.1, 6.8]
            fig = go.Figure(go.Bar(
                x=models, y=rmse_v,
                marker_color=[GREEN if i == 2 else BLUE for i in range(4)],
                text=[f"{v}" for v in rmse_v], textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Model", yaxis_title="Val RMSE (lower = better)")
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown(seg_header("Validation R² Comparison"), unsafe_allow_html=True)
            r2_v = [0.48, 0.61, 0.84, 0.81]
            fig = go.Figure(go.Bar(
                x=models, y=r2_v,
                marker_color=[GREEN if i == 2 else BLUE for i in range(4)],
                text=[f"{v}" for v in r2_v], textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.add_hline(y=0.80, line_color=YELLOW, line_dash="dash",
                          annotation_text="R²=0.80 target", annotation_font_color=YELLOW)
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Model", yaxis_title="Val R² (higher = better)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(seg_header("Why GBT Outperforms Linear Regression"), unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#13161d;border:1px solid #252b3b;border-radius:8px;padding:16px 20px;font-size:.87rem;line-height:1.7;color:#e8ecf4'>
        <b style='color:#f5c518'>Zero-demand problem:</b> Most zone-hours have zero trips. Linear regression tries to fit a line through zeros and real demand simultaneously — it can't do both well.<br><br>
        <b style='color:#f5c518'>Non-linearity:</b> Demand isn't linearly related to lag features. During special events (NYE, blizzard recovery), the relationship changes entirely. GBT captures these non-linear breakpoints via tree splits.<br><br>
        <b style='color:#f5c518'>Interaction effects:</b> High lag_1 during Friday evening means something different than high lag_1 at 3 AM on Sunday. GBT naturally captures these interaction effects through deep tree splits.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 5: GenAI Q&A ──────────────────────────────────────────────────────
    with tab_genai:
        genai_panel(
            "demand", DEMAND_SYS,
            ["When is demand highest during the week?",
             "What indicators signal demand spikes?",
             "Why can't we show a geographic demand map?"],
        )
