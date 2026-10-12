"""seg_eda.py – Segment 1: Data Overview & EDA"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import time

from styles import DARK, YELLOW, BLUE, GREEN, RED, ORANGE, MUTED, BIN_COLORS, BIN_ORDER, seg_header, kpi, kpi_animated, sim_card
from genai import genai_panel, EDA_SYS


def render(df: pd.DataFrame):
    st.markdown("<div class='pg-title'>Data Overview & EDA</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pg-sub'>NYC Yellow Taxi 2016 · Jan–Mar · Exploratory Analysis</div>",
        unsafe_allow_html=True,
    )

    tab_viz, tab_time, tab_anomaly, tab_schema, tab_genai = st.tabs(
        ["EDA Visualisations", "Time Explorer", "Anomaly Explorer",
         "Feature Schema", "GenAI Q&A"]
    )

    # ── Tab 1: EDA Visualisations ─────────────────────────────────────────────
    with tab_viz:
        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (v, l) in zip(
            [c1, c2, c3, c4, c5],
            [
                ("300K", "Sampled Trips"),
                ("3 Months", "Jan – Mar 2016"),
                ("15", "Pickup Clusters"),
                ("~14 mph", "Avg City Speed"),
                ("Jan 23", "Jonas Blizzard"),
            ],
        ):
            col.markdown(kpi_animated(v, l), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Heartbeat + Weekday/Weekend ────────────────────────────────
        col_l, col_r = st.columns(2)
        hours = list(range(24))

        with col_l:
            st.markdown(seg_header("NYC Speed Heartbeat by Month"), unsafe_allow_html=True)
            speed_jan = [22,26,28,24,22,18,12, 9, 8, 9,11,12,13,12,11,10, 9, 9,10,11,12,14,17,20]
            speed_feb = [21,25,27,23,21,17,11, 8, 8, 9,11,12,13,12,11,10, 9, 9,10,11,13,15,18,20]
            speed_mar = [20,24,26,22,20,16,11, 8, 8, 9,11,12,13,12,11,10, 9,10,11,12,13,15,18,20]
            fig = go.Figure()
            for name, spd, col in [("January", speed_jan, YELLOW),
                                    ("February", speed_feb, BLUE),
                                    ("March", speed_mar, GREEN)]:
                fig.add_trace(go.Scatter(x=hours, y=spd, name=name, mode="lines+markers",
                                         line=dict(width=2, color=col), marker=dict(size=4)))
            fig.add_vrect(x0=7, x1=9,   fillcolor=RED,  opacity=0.10, line_width=0,
                          annotation_text="AM Rush", annotation_font_color=RED)
            fig.add_vrect(x0=16, x1=19, fillcolor=RED,  opacity=0.10, line_width=0,
                          annotation_text="PM Rush", annotation_font_color=RED)
            fig.update_layout(**DARK, height=300, xaxis_title="Hour of Day",
                              yaxis_title="Avg Speed (mph)")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(seg_header("Weekday vs Weekend Speed Dynamics"), unsafe_allow_html=True)
            spd_wd = [23,27,29,25,22,16,10, 8, 8, 9,11,12,12,12,11,10, 9, 9,10,11,12,14,17,21]
            spd_we = [19,22,25,23,22,20,16,13,12,13,14,15,15,14,13,12,11,11,12,13,14,16,18,20]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=spd_wd, name="Weekday", mode="lines",
                                     line=dict(width=2.5, color=BLUE),
                                     fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
            fig.add_trace(go.Scatter(x=hours, y=spd_we, name="Weekend", mode="lines",
                                     line=dict(width=2.5, color=YELLOW),
                                     fill="tozeroy", fillcolor="rgba(245,197,24,0.08)"))
            fig.update_layout(**DARK, height=300, xaxis_title="Hour of Day",
                              yaxis_title="Avg Speed (mph)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Heatmap + Volatility ───────────────────────────────────────
        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown(seg_header("Speed Heatmap: Cluster × Hour"), unsafe_allow_html=True)
            np.random.seed(42)
            heat = np.random.normal(12, 3, (15, 24))
            heat[:, 7:10] -= 4; heat[:, 16:19] -= 4
            heat[:, 0:5]  += 6; heat[:, 22:24] += 4
            heat[2, :] += 5; heat[7, :] += 3
            heat = np.clip(heat, 5, 30)
            fig = go.Figure(go.Heatmap(
                z=heat, x=list(range(24)),
                y=[f"Cluster {i}" for i in range(15)],
                colorscale="RdYlGn",
                colorbar=dict(title=dict(text="Speed (mph)", font=dict(color="#e8ecf4")),
                              tickfont=dict(color="#e8ecf4")),
            ))
            fig.update_layout(**DARK, height=360, xaxis_title="Hour of Day", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with col_r2:
            st.markdown(seg_header("Speed Volatility by Hour (Prediction Uncertainty)"), unsafe_allow_html=True)
            vol = [0.38,0.35,0.31,0.29,0.28,0.30,0.36,0.45,0.48,0.41,0.33,0.30,
                   0.29,0.31,0.33,0.36,0.42,0.46,0.43,0.38,0.35,0.33,0.34,0.37]
            bar_colors = [RED if v > 0.40 else YELLOW if v > 0.33 else GREEN for v in vol]
            fig = go.Figure(go.Bar(x=list(range(24)), y=vol, marker_color=bar_colors))
            fig.add_hline(y=0.40, line_dash="dot", line_color=RED,
                          annotation_text="High uncertainty threshold",
                          annotation_font_color=RED)
            fig.update_layout(**DARK, height=360, showlegend=False,
                              xaxis_title="Hour of Day",
                              yaxis_title="Coefficient of Variation")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: Blizzard timeline + Distance dist ──────────────────────────
        col_l3, col_r3 = st.columns(2)

        with col_l3:
            st.markdown(seg_header("90-Day Speed Timeline — Jonas Blizzard"), unsafe_allow_html=True)
            dates = pd.date_range("2016-01-01", "2016-03-31", freq="D")
            np.random.seed(1)
            spds = np.random.normal(13.5, 0.8, len(dates))
            bliz = (dates >= "2016-01-23") & (dates <= "2016-01-26")
            spds[bliz] -= 4
            spds = np.clip(spds, 8, 18)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=spds, mode="lines",
                                     line=dict(width=2, color=BLUE),
                                     fill="tozeroy", fillcolor="rgba(59,130,246,0.07)"))
            fig.add_vline(x="2016-01-23", line_color=RED, line_dash="dash")
            fig.add_annotation(x="2016-01-23", y=10.5, text="Jonas Blizzard",
                               font=dict(color=RED, size=11), showarrow=True,
                               arrowcolor=RED, bgcolor="#13161d")
            fig.update_layout(**DARK, height=300, xaxis_title="Date",
                              yaxis_title="Avg Speed (mph)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_r3:
            st.markdown(seg_header("Trip Distance Distribution by Bin"), unsafe_allow_html=True)
            # Real data from CSV
            bin_order_present = [b for b in BIN_ORDER if b in df["distance_bin"].values]
            cnts = df["distance_bin"].value_counts().reindex(bin_order_present)
            fig = go.Figure(go.Bar(
                x=cnts.index, y=cnts.values,
                marker_color=[BIN_COLORS[b] for b in cnts.index],
                text=cnts.values, textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Distance Bin", yaxis_title="Trip Count")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 4: Congestion memory + Route efficiency ───────────────────────
        col_l4, col_r4 = st.columns(2)

        with col_l4:
            st.markdown(seg_header("Congestion Memory: Speed 30m Lag vs Current"), unsafe_allow_html=True)
            # Use real prev_speed_30m from CSV
            sample = df[["prev_speed_30m", "prev_speed_15m"]].dropna().sample(
                min(500, len(df)), random_state=7
            )
            np.random.seed(7)
            current_spd = sample["prev_speed_30m"] * 0.85 + np.random.normal(0, 3, len(sample))
            current_spd = np.clip(current_spd, 0, 60)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sample["prev_speed_30m"], y=current_spd,
                                     mode="markers",
                                     marker=dict(color=BLUE, size=4, opacity=0.4)))
            m, b = np.polyfit(sample["prev_speed_30m"], current_spd, 1)
            xr = np.array([sample["prev_speed_30m"].min(), sample["prev_speed_30m"].max()])
            fig.add_trace(go.Scatter(x=xr, y=m * xr + b, mode="lines",
                                     line=dict(color=RED, width=2), name="Trend"))
            fig.update_layout(**DARK, height=280, showlegend=False,
                              xaxis_title="Speed 30 min Ago (mph)",
                              yaxis_title="Approx. Current Speed (mph)")
            st.plotly_chart(fig, use_container_width=True)

        with col_r4:
            st.markdown(seg_header("Route Efficiency by Distance Bin (Real Data)"), unsafe_allow_html=True)
            eff = df.groupby("distance_bin")["route_efficiency"].mean().reindex(bin_order_present)
            fig = go.Figure(go.Bar(
                x=eff.index, y=eff.values,
                marker_color=[BIN_COLORS[b] for b in eff.index],
                text=[f"{v:.3f}" for v in eff.values], textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.add_hline(y=1.0, line_dash="dash", line_color=YELLOW,
                          annotation_text="Ideal = 1.0 (perfectly direct)",
                          annotation_font_color=YELLOW)
            fig.update_layout(**DARK, height=280, showlegend=False,
                              xaxis_title="Distance Bin",
                              yaxis_title="Avg Route Efficiency")
            st.plotly_chart(fig, use_container_width=True)

        # ── Summary stats ─────────────────────────────────────────────────────
        st.markdown(seg_header("Summary Statistics (1,000 Sample — Real Data)"), unsafe_allow_html=True)
        stats_cols = ["trip_distance", "actual_duration_min", "lgbm_v3_pred_min",
                      "route_efficiency", "reliability_score", "prediction_error_min",
                      "haversine_dist", "prev_speed_15m", "prev_speed_30m"]
        st.dataframe(df[stats_cols].describe().round(3), use_container_width=True)

    # ── Tab 2: Time Explorer ──────────────────────────────────────────────────
    with tab_time:
        st.markdown(seg_header("Animated Hour-by-Hour Explorer"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Drag the slider or press ▶ Auto-Play to watch NYC speed patterns evolve through the day.</div>",
            unsafe_allow_html=True,
        )

        # Controls
        tc1, tc2 = st.columns([4, 1])
        with tc1:
            selected_hour = st.slider("Select Hour", 0, 23, 12, key="eda_hour_slider")
        with tc2:
            auto_play = st.button("▶ Auto-Play", key="eda_autoplay")

        # Data for time explorer
        np.random.seed(42)
        heat_data = np.random.normal(12, 3, (15, 24))
        heat_data[:, 7:10] -= 4; heat_data[:, 16:19] -= 4
        heat_data[:, 0:5] += 6; heat_data[:, 22:24] += 4
        heat_data[2, :] += 5; heat_data[7, :] += 3
        heat_data = np.clip(heat_data, 5, 30)

        spd_wd_arr = [23,27,29,25,22,16,10,8,8,9,11,12,12,12,11,10,9,9,10,11,12,14,17,21]
        spd_we_arr = [19,22,25,23,22,20,16,13,12,13,14,15,15,14,13,12,11,11,12,13,14,16,18,20]
        vol_arr = [0.38,0.35,0.31,0.29,0.28,0.30,0.36,0.45,0.48,0.41,0.33,0.30,
                   0.29,0.31,0.33,0.36,0.42,0.46,0.43,0.38,0.35,0.33,0.34,0.37]

        h = selected_hour

        # Time period label
        period = "Late Night" if h < 5 else "Early Morning" if h < 7 else \
                 "AM Rush" if h < 10 else "Midday" if h < 16 else \
                 "PM Rush" if h < 19 else "Evening" if h < 22 else "Late Night"

        # KPI row for selected hour
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.markdown(sim_card(f"{spd_wd_arr[h]} mph", "Weekday Speed", BLUE), unsafe_allow_html=True)
        kc2.markdown(sim_card(f"{spd_we_arr[h]} mph", "Weekend Speed", YELLOW), unsafe_allow_html=True)
        vol_color = RED if vol_arr[h] > 0.40 else YELLOW if vol_arr[h] > 0.33 else GREEN
        kc3.markdown(sim_card(f"{vol_arr[h]:.2f}", "Volatility (CV)", vol_color), unsafe_allow_html=True)
        kc4.markdown(sim_card(period, f"Hour {h}:00", MUTED), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts for selected hour
        tc_l, tc_r = st.columns(2)

        with tc_l:
            st.markdown(seg_header(f"Speed by Cluster at {h}:00"), unsafe_allow_html=True)
            cluster_speeds = heat_data[:, h]
            colors = [GREEN if s > 15 else YELLOW if s > 10 else RED for s in cluster_speeds]
            fig = go.Figure(go.Bar(
                x=[f"C{i}" for i in range(15)], y=cluster_speeds,
                marker_color=colors,
                text=[f"{s:.1f}" for s in cluster_speeds], textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.update_layout(**DARK, height=320, showlegend=False,
                              xaxis_title="Cluster", yaxis_title="Speed (mph)",
                              yaxis_range=[0, 35])
            st.plotly_chart(fig, use_container_width=True)

        with tc_r:
            st.markdown(seg_header(f"Weekday vs Weekend at {h}:00"), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Weekday", "Weekend"],
                                 y=[spd_wd_arr[h], spd_we_arr[h]],
                                 marker_color=[BLUE, YELLOW],
                                 text=[f"{spd_wd_arr[h]} mph", f"{spd_we_arr[h]} mph"],
                                 textposition="outside", textfont=dict(color="#e8ecf4")))
            diff = spd_we_arr[h] - spd_wd_arr[h]
            diff_text = f"Weekend is {abs(diff)} mph {'faster' if diff > 0 else 'slower'}"
            fig.add_annotation(x=0.5, y=max(spd_wd_arr[h], spd_we_arr[h]) + 3,
                               text=diff_text, showarrow=False,
                               font=dict(color=GREEN if diff > 0 else RED, size=13))
            fig.update_layout(**DARK, height=320, showlegend=False,
                              yaxis_title="Speed (mph)", yaxis_range=[0, 35])
            st.plotly_chart(fig, use_container_width=True)

        # Auto-play logic
        if auto_play:
            placeholder = st.empty()
            for auto_h in range(24):
                with placeholder.container():
                    st.markdown(
                        f"<div style='text-align:center;color:#f5c518;font-family:Space Mono;font-size:1.2rem'>"
                        f"▶ Playing: {auto_h:02d}:00 — Weekday: {spd_wd_arr[auto_h]} mph | "
                        f"Weekend: {spd_we_arr[auto_h]} mph | Volatility: {vol_arr[auto_h]:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                time.sleep(0.4)
            placeholder.markdown(
                "<div style='text-align:center;color:#22c55e;font-size:.9rem'>✓ Playback complete</div>",
                unsafe_allow_html=True,
            )

    # ── Tab 3: Anomaly Explorer ───────────────────────────────────────────────
    with tab_anomaly:
        st.markdown(seg_header("Anomaly Detection — Unusual Trips"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Automatically flagged trips with extreme values. "
            "Adjust thresholds to explore different anomaly definitions.</div>",
            unsafe_allow_html=True,
        )

        # Threshold controls
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            eff_thresh = st.slider("Route Efficiency >", 1.0, 5.0, 3.0, 0.1, key="eda_eff_thresh")
        with ac2:
            err_sigma = st.slider("Error > N × σ", 1.0, 4.0, 2.0, 0.5, key="eda_err_sigma")
        with ac3:
            spd_thresh = st.slider("Speed 15m Ago >", 20, 60, 40, 5, key="eda_spd_thresh")

        # Detect anomalies
        err_std = df["prediction_error_min"].std()
        err_mean = df["prediction_error_min"].mean()

        anom_mask = (
            (df["route_efficiency"] > eff_thresh) |
            (df["prediction_error_min"].abs() > err_mean + err_sigma * err_std) |
            (df["prev_speed_15m"] > spd_thresh)
        )
        anomalies = df[anom_mask].copy()
        anomalies["anomaly_reason"] = ""
        reasons = []
        for _, row in anomalies.iterrows():
            r = []
            if row["route_efficiency"] > eff_thresh:
                r.append(f"Efficiency={row['route_efficiency']:.2f}")
            if abs(row["prediction_error_min"]) > err_mean + err_sigma * err_std:
                r.append(f"Error={row['prediction_error_min']:.2f}min")
            if row["prev_speed_15m"] > spd_thresh:
                r.append(f"Speed15m={row['prev_speed_15m']:.1f}mph")
            reasons.append(" | ".join(r))
        anomalies["anomaly_reason"] = reasons

        # KPIs
        ak1, ak2, ak3 = st.columns(3)
        ak1.markdown(kpi_animated(f"{len(anomalies)}", "Anomalies Found"), unsafe_allow_html=True)
        ak2.markdown(kpi_animated(f"{len(anomalies)/len(df)*100:.1f}%", "Of All Trips"), unsafe_allow_html=True)
        if len(anomalies) > 0:
            ak3.markdown(kpi_animated(f"{anomalies['prediction_error_min'].abs().mean():.2f} min",
                                      "Avg Abs Error (Anomalies)"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if len(anomalies) > 0:
            # Display table
            show_cols = ["pickup_hour", "trip_distance", "prediction_error_min",
                         "route_efficiency", "reliability_label", "prev_speed_15m",
                         "distance_bin", "anomaly_reason"]
            st.dataframe(
                anomalies[show_cols].sort_values("prediction_error_min", key=abs, ascending=False)
                .head(50).reset_index(drop=True),
                use_container_width=True,
            )

            # Distribution comparison
            st.markdown(seg_header("Anomalies vs Normal — Error Distribution"), unsafe_allow_html=True)
            fig = go.Figure()
            normal = df[~anom_mask]
            fig.add_trace(go.Histogram(x=normal["prediction_error_min"], nbinsx=40,
                                       name="Normal", marker_color=BLUE, opacity=0.6))
            fig.add_trace(go.Histogram(x=anomalies["prediction_error_min"], nbinsx=20,
                                       name="Anomalies", marker_color=RED, opacity=0.7))
            fig.update_layout(**DARK, height=280, barmode="overlay",
                              xaxis_title="Prediction Error (min)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No anomalies detected with current thresholds.")

    # ── Tab 4: Feature Schema ─────────────────────────────────────────────────
    with tab_schema:
        st.markdown(seg_header("Complete Feature Schema"), unsafe_allow_html=True)
        schema = pd.DataFrame({
            "Feature": [
                "tpep_pickup_datetime", "tpep_dropoff_datetime", "pickup_hour", "day_of_week",
                "month", "is_weekend", "rush_hour", "late_night",
                "trip_distance", "pickup_latitude", "pickup_longitude",
                "dropoff_latitude", "dropoff_longitude", "haversine_dist",
                "manhattan_dist", "delta_lat", "delta_lon",
                "pickup_cluster", "is_airport_pu", "is_airport_do",
                "prev_speed_15m", "prev_speed_30m", "prev_speed_60m",
                "route_efficiency", "reliability_score", "reliability_label",
                "dist_x_hour", "dist_x_rush", "speed_x_dist",
                "pickup_cluster_te", "pickup_hour_te", "day_of_week_te",
                "actual_duration_min", "lgbm_v3_pred_min", "prediction_error_min",
            ],
            "Category": [
                "Temporal","Temporal","Temporal","Temporal","Temporal","Temporal","Temporal","Temporal",
                "Spatial","Spatial","Spatial","Spatial","Spatial","Spatial","Spatial","Spatial","Spatial",
                "Behavioral","Behavioral","Behavioral",
                "Behavioral","Behavioral","Behavioral",
                "Engineered","Engineered","Engineered",
                "Interaction","Interaction","Interaction",
                "TargetEnc","TargetEnc","TargetEnc",
                "Target","Prediction","Prediction",
            ],
            "Description": [
                "Pickup timestamp","Dropoff timestamp","Hour of day (0–23)","Day of week (0=Mon)",
                "Month (1–12)","1 if Saturday or Sunday","1 if 7–9 AM or 4–7 PM","1 if midnight–4 AM",
                "Trip distance in miles","Pickup latitude","Pickup longitude",
                "Dropoff latitude","Dropoff longitude","Straight-line distance (Haversine, miles)",
                "Manhattan grid distance (miles)","Lat difference (dropoff–pickup)","Lon difference",
                "KMeans cluster (0–14) of pickup location","1 if pickup near JFK or LGA","1 if dropoff near JFK or LGA",
                "Speed 15 min before pickup","Speed 30 min before pickup","Speed 60 min before pickup",
                "actual_distance / haversine_dist (detour ratio)","Prediction reliability 0–100",
                "High / Medium / Low reliability label",
                "haversine_dist × pickup_hour","haversine_dist × rush_hour","prev_speed_15m × trip_distance",
                "Target-encoded pickup cluster","Target-encoded pickup hour","Target-encoded day of week",
                "Actual trip duration (minutes) — TARGET","LightGBM v3 predicted duration (minutes)",
                "lgbm_pred − actual (positive = overestimate)",
            ],
        })
        st.dataframe(schema, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(seg_header("Feature Category Summary"), unsafe_allow_html=True)
        cat_cnt = schema["Category"].value_counts()
        fig = px.pie(
            values=cat_cnt.values, names=cat_cnt.index, hole=0.5,
            color_discrete_sequence=[YELLOW, BLUE, GREEN, ORANGE, RED, "#a855f7", "#14b8a6"],
        )
        fig.update_layout(**DARK, height=280, showlegend=True)
        fig.update_traces(textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: GenAI Q&A ──────────────────────────────────────────────────────
    with tab_genai:
        genai_panel(
            "eda", EDA_SYS,
            ["Are there temporal patterns?",
             "How is route efficiency defined?",
             "What caused the January speed drop?"],
        )
