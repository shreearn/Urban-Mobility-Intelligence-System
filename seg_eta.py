"""seg_eta.py – Segment 2: ETA Prediction"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles import (DARK, YELLOW, BLUE, GREEN, RED, MUTED, BIN_COLORS, BIN_ORDER,
                    seg_header, kpi, kpi_animated, pill, sim_card)
from genai import genai_panel, ETA_SYS


def _compute_reliability(distance_mi, is_airport, rush_hour, hour, cluster, is_weekend):
    """Replicate the rule-based reliability score (0-100)."""
    score = 100
    if distance_mi > 20:
        score -= 40
    elif distance_mi > 10:
        score -= 25
    elif distance_mi > 5:
        score -= 10
    if is_airport:
        score -= 30
    if rush_hour:
        score -= 10
    if 13 <= hour <= 17:
        score -= 10
    if cluster in [1, 3]:
        score -= 15
    if is_weekend:
        score += 5
    return max(0, min(100, score))


def render(df: pd.DataFrame, is_full_parquet: bool):
    st.markdown("<div class='pg-title'>ETA Prediction</div>", unsafe_allow_html=True)

    src = f"Full Parquet ({len(df):,} trips)" if is_full_parquet else f"Sample CSV ({len(df):,} trips)"
    st.markdown(
        f"<div class='pg-sub'>LightGBM v3 · March 2016 · Source: {src}</div>",
        unsafe_allow_html=True,
    )
    if not is_full_parquet:
        st.info(
            "📂 **Full parquet not found.** Using 1,000-row CSV sample. "
            "Place `taxi_2016_test_final_with_predictions.parquet` in the `data/` folder "
            "to unlock the full 11.9M-trip analysis.",
            icon="ℹ️",
        )

    tab_viz, tab_model, tab_sim, tab_outlier, tab_lookup, tab_genai = st.tabs(
        ["Visualisations", "Model Comparison", "What-If Simulator",
         "Outlier Trips", "Trip Lookup", "GenAI Q&A"]
    )

    # ── Pre-compute ───────────────────────────────────────────────────────────
    mae       = df["prediction_error_min"].abs().mean()
    within2   = (df["prediction_error_min"].abs() <= 2).mean() * 100
    avg_rel   = df["reliability_score"].mean()
    corr_val  = df[["lgbm_v3_pred_min", "actual_duration_min"]].corr().iloc[0, 1]
    rmse      = np.sqrt((df["prediction_error_min"] ** 2).mean())

    # ── Tab 1: Visualisations ─────────────────────────────────────────────────
    with tab_viz:
        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (v, l) in zip(
            [c1, c2, c3, c4, c5],
            [
                (f"{mae:.2f} min", "Mean Abs Error"),
                (f"{rmse:.2f} min", "RMSE"),
                (f"{within2:.1f}%", "Within 2-min"),
                (f"{avg_rel:.1f}", "Avg Reliability"),
                (f"{corr_val:.4f}", "Pred-Actual Corr"),
            ],
        ):
            col.markdown(kpi_animated(v, l), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Scatter + Error hist ───────────────────────────────────────
        cl, cr = st.columns(2)
        with cl:
            st.markdown(seg_header("Predicted vs Actual Duration"), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["actual_duration_min"], y=df["lgbm_v3_pred_min"],
                mode="markers",
                marker=dict(color=df["reliability_score"], colorscale="RdYlGn",
                            size=5, opacity=0.65,
                            colorbar=dict(title=dict(text="Reliability", font=dict(color="#e8ecf4")),
                                          tickfont=dict(color="#e8ecf4"))),
                name="Trips",
            ))
            mx = max(df["actual_duration_min"].max(), df["lgbm_v3_pred_min"].max())
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                                     line=dict(color=YELLOW, dash="dash", width=2),
                                     name="Perfect Prediction"))
            fig.update_layout(**DARK, height=340,
                              xaxis_title="Actual Duration (min)",
                              yaxis_title="Predicted Duration (min)")
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown(seg_header("Prediction Error Distribution"), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df["prediction_error_min"], nbinsx=55,
                                       marker_color=BLUE, opacity=0.8))
            fig.add_vline(x=0, line_color=YELLOW, line_dash="dash",
                          annotation_text="Zero error", annotation_font_color=YELLOW)
            mean_err = df["prediction_error_min"].mean()
            fig.add_vline(x=mean_err, line_color=GREEN, line_dash="dot",
                          annotation_text=f"Mean={mean_err:.2f}",
                          annotation_font_color=GREEN)
            fig.update_layout(**DARK, height=340, showlegend=False,
                              xaxis_title="Prediction Error (min)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Error by hour + Reliability dist ───────────────────────────
        cl2, cr2 = st.columns(2)
        with cl2:
            st.markdown(seg_header("Mean Error by Hour of Day"), unsafe_allow_html=True)
            hourly = df.groupby("pickup_hour")["prediction_error_min"].mean().reset_index()
            bar_c = [RED if v > 1 else GREEN if v < -1 else YELLOW
                     for v in hourly["prediction_error_min"]]
            fig = go.Figure(go.Bar(x=hourly["pickup_hour"],
                                   y=hourly["prediction_error_min"],
                                   marker_color=bar_c))
            fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.4)
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Hour of Day",
                              yaxis_title="Mean Error (min)")
            st.plotly_chart(fig, use_container_width=True)

        with cr2:
            st.markdown(seg_header("Reliability Label Distribution"), unsafe_allow_html=True)
            rel_cnt = df["reliability_label"].value_counts()
            rel_c = {"High": GREEN, "Medium": YELLOW, "Low": RED}
            fig = go.Figure(go.Bar(
                x=rel_cnt.index, y=rel_cnt.values,
                marker_color=[rel_c.get(l, BLUE) for l in rel_cnt.index],
                text=rel_cnt.values, textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Reliability", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: Rush hour + Distance bin error ─────────────────────────────
        cl3, cr3 = st.columns(2)
        with cl3:
            st.markdown(seg_header("Error Distribution: Rush vs Non-Rush"), unsafe_allow_html=True)
            fig = go.Figure()
            for flag, name, color in [(0, "Non-Rush", GREEN), (1, "Rush Hour", RED)]:
                sub = df[df["rush_hour"] == flag]["prediction_error_min"]
                fig.add_trace(go.Box(y=sub, name=name, marker_color=color, boxmean=True))
            fig.update_layout(**DARK, height=300, showlegend=False,
                              yaxis_title="Prediction Error (min)")
            st.plotly_chart(fig, use_container_width=True)

        with cr3:
            st.markdown(seg_header("Mean Absolute Error by Distance Bin"), unsafe_allow_html=True)
            bins_present = [b for b in BIN_ORDER if b in df["distance_bin"].values]
            bin_mae = df.groupby("distance_bin")["prediction_error_min"].apply(
                lambda x: x.abs().mean()
            ).reindex(bins_present)
            fig = go.Figure(go.Bar(
                x=bin_mae.index, y=bin_mae.values,
                marker_color=[BIN_COLORS[b] for b in bin_mae.index],
                text=[f"{v:.2f}" for v in bin_mae.values], textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Distance Bin", yaxis_title="MAE (min)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 4: Reliability score histogram ────────────────────────────────
        st.markdown(seg_header("Reliability Score Distribution (0–100)"), unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=df["reliability_score"], nbinsx=30,
                                     marker_color=YELLOW, opacity=0.85))
        fig.add_vline(x=75, line_color=GREEN, line_dash="dash",
                      annotation_text="High threshold (75)", annotation_font_color=GREEN)
        fig.add_vline(x=50, line_color=RED, line_dash="dash",
                      annotation_text="Medium threshold (50)", annotation_font_color=RED)
        fig.update_layout(**DARK, height=260, showlegend=False,
                          xaxis_title="Reliability Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Model Comparison ───────────────────────────────────────────────
    with tab_model:
        st.markdown(seg_header("Model Performance — All Experiments"), unsafe_allow_html=True)
        model_df = pd.DataFrame({
            "Model":       ["LightGBM v1","LightGBM v2","LightGBM v3 ✓",
                            "XGBoost (5M)","CatBoost (5M)","GAM (200k)"],
            "Data Size":   ["21.7M","21.7M","21.7M","5M","5M","200k"],
            "Rounds":      ["1000","2000","3000","1000","1000","gridsearch"],
            "MAE (min)":   [2.393, 2.335, 2.306, 2.700, 2.744, 3.138],
            "R²":          [0.852, 0.858, 0.861, 0.828, 0.823, 0.699],
            "MAE % mean":  ["17.5%","17.0%","16.8%","19.7%","20.0%","22.9%"],
            "Notes":       ["Baseline","2k rounds","CHOSEN — best","Fair comparison",
                            "Slightly weaker","Interpretable but weak"],
        })
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            st.markdown(seg_header("MAE Comparison (5M Fair Sample)"), unsafe_allow_html=True)
            models_s = ["LightGBM (5M)", "XGBoost (5M)", "CatBoost (5M)", "GAM (200k)"]
            mae_s    = [2.684, 2.700, 2.744, 3.138]
            colors_m = [GREEN, BLUE, YELLOW, RED]
            fig = go.Figure(go.Bar(x=mae_s[::-1], y=models_s[::-1], orientation="h",
                                   marker_color=colors_m[::-1],
                                   text=[f"{v:.3f}" for v in mae_s[::-1]],
                                   textposition="outside",
                                   textfont=dict(color="#e8ecf4")))
            fig.add_vline(x=min(mae_s), line_color=GREEN, line_dash="dot")
            fig.update_layout(**DARK, height=280, showlegend=False,
                              xaxis_title="MAE (minutes)",
                              xaxis_range=[2.5, 3.3])
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown(seg_header("Reliability Score — How It Works"), unsafe_allow_html=True)
            rules = pd.DataFrame({
                "Condition":    ["Base score","dist > 20 mi","dist > 10 mi","dist > 5 mi",
                                 "Airport trip","Rush hour","Hours 13–17",
                                 "Clusters 1 or 3","Weekend","Final clamp"],
                "Points":       ["+100","-40","-25","-10","-30","-10","-10","-15","+5","0–100"],
                "Reason":       ["Start perfect","Very unpredictable","Long trips noisier",
                                 "Medium noise","JFK/LGA 2× worse MAE","Higher variance",
                                 "Afternoon error spike","Worst-performing zones",
                                 "Easier to predict","Cannot exceed 100"],
            })
            st.dataframe(rules, use_container_width=True, hide_index=True)

        st.markdown(seg_header("Feature Importance Top 15 (LightGBM Gain)"), unsafe_allow_html=True)
        fi_feat = ["trip_distance","haversine_dist","dropoff_latitude","manhattan_dist",
                   "pickup_hour_te","pickup_latitude","dropoff_longitude","pickup_longitude",
                   "delta_lat","pickup_cluster_te","day_of_week_te","delta_lon",
                   "prev_speed_15m","prev_speed_30m","prev_speed_60m"]
        fi_imp  = [4850,1810,680,520,490,430,380,310,270,240,200,170,120,100,80]
        fig = go.Figure(go.Bar(
            x=fi_imp[::-1], y=fi_feat[::-1], orientation="h",
            marker=dict(color=fi_imp[::-1], colorscale=[[0, BLUE],[1, YELLOW]]),
        ))
        fig.update_layout(**DARK, height=380, showlegend=False,
                          xaxis_title="Importance (Gain)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: What-If Simulator ──────────────────────────────────────────────
    with tab_sim:
        st.markdown(seg_header("What-If ETA Simulator"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Adjust trip parameters to see how the predicted ETA and reliability change. "
            "Uses pre-aggregated statistics from 11.9M real trips.</div>",
            unsafe_allow_html=True,
        )

        # Load lookup table
        from data_loader import load_eta_lookup
        lookup = load_eta_lookup()

        # Controls
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sim_distance = st.slider("Trip Distance (miles)", 0.5, 30.0, 5.0, 0.5, key="eta_sim_dist")
            sim_hour = st.selectbox("Pickup Hour", list(range(24)), index=12, key="eta_sim_hour")
        with sc2:
            sim_cluster = st.selectbox("Pickup Cluster", list(range(15)), index=0, key="eta_sim_cluster")
            sim_rush = st.toggle("Rush Hour (7-9 AM / 4-7 PM)", key="eta_sim_rush")
        with sc3:
            sim_weekend = st.toggle("Weekend", key="eta_sim_weekend")
            sim_airport = st.toggle("Airport Trip", key="eta_sim_airport")

        # Determine distance bin
        dist_km = sim_distance * 1.60934
        if dist_km <= 2:
            sim_bin = "Short (0-2 km)"
        elif dist_km <= 5:
            sim_bin = "Medium (2-5 km)"
        elif dist_km <= 10:
            sim_bin = "Long (5-10 km)"
        else:
            sim_bin = "Very Long (10+ km)"

        # Compute reliability
        rel_score = _compute_reliability(
            sim_distance, sim_airport, sim_rush, sim_hour, sim_cluster, sim_weekend
        )
        rel_label = "High" if rel_score >= 75 else "Medium" if rel_score >= 50 else "Low"

        # Look up prediction
        match = lookup[
            (lookup["pickup_hour"] == sim_hour) &
            (lookup["pickup_cluster"] == sim_cluster) &
            (lookup["distance_bin"] == sim_bin) &
            (lookup["rush_hour"] == int(sim_rush)) &
            (lookup["is_weekend"] == int(sim_weekend))
        ]

        if len(match) > 0:
            row = match.iloc[0]
            pred_eta = row["avg_pred"]
            actual_eta = row["avg_actual"]
            sim_mae = row["mae"]
            sim_std = row["std_pred"]
            sim_count = int(row["count"])
        else:
            # Fallback: looser match (just hour + cluster + bin)
            match2 = lookup[
                (lookup["pickup_hour"] == sim_hour) &
                (lookup["pickup_cluster"] == sim_cluster) &
                (lookup["distance_bin"] == sim_bin)
            ]
            if len(match2) > 0:
                pred_eta = match2["avg_pred"].mean()
                actual_eta = match2["avg_actual"].mean()
                sim_mae = match2["mae"].mean()
                sim_std = match2["std_pred"].mean()
                sim_count = int(match2["count"].sum())
            else:
                # Fallback: just hour + bin
                match3 = lookup[
                    (lookup["pickup_hour"] == sim_hour) &
                    (lookup["distance_bin"] == sim_bin)
                ]
                if len(match3) > 0:
                    pred_eta = match3["avg_pred"].mean()
                    actual_eta = match3["avg_actual"].mean()
                    sim_mae = match3["mae"].mean()
                    sim_std = match3["std_pred"].mean()
                    sim_count = int(match3["count"].sum())
                else:
                    pred_eta = sim_distance * 3.5
                    actual_eta = pred_eta * 1.04
                    sim_mae = 2.3
                    sim_std = 3.0
                    sim_count = 0

        st.markdown("<br>", unsafe_allow_html=True)

        # Results display
        rc1, rc2, rc3, rc4 = st.columns(4)
        color_eta = GREEN if pred_eta < 15 else YELLOW if pred_eta < 30 else RED
        rc1.markdown(sim_card(f"{pred_eta:.1f} min", "Predicted ETA", color_eta,
                              f"±{sim_mae:.1f} min expected error"), unsafe_allow_html=True)
        rc2.markdown(sim_card(f"{actual_eta:.1f} min", "Avg Actual Duration", BLUE), unsafe_allow_html=True)
        rel_color = GREEN if rel_label == "High" else YELLOW if rel_label == "Medium" else RED
        rc3.markdown(sim_card(f"{rel_score}", "Reliability Score", rel_color,
                              f"Label: {rel_label}"), unsafe_allow_html=True)
        rc4.markdown(sim_card(f"{sim_count:,}", "Matching Trips", MUTED,
                              f"Bin: {sim_bin}"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence range visualization
        st.markdown(seg_header("Confidence Range"), unsafe_allow_html=True)
        fig = go.Figure()
        low = max(0, pred_eta - sim_std)
        high = pred_eta + sim_std
        fig.add_trace(go.Bar(x=["Optimistic", "Predicted", "Pessimistic"],
                             y=[low, pred_eta, high],
                             marker_color=[GREEN, YELLOW, RED],
                             text=[f"{low:.1f}", f"{pred_eta:.1f}", f"{high:.1f}"],
                             textposition="outside", textfont=dict(color="#e8ecf4")))
        fig.update_layout(**DARK, height=250, showlegend=False,
                          yaxis_title="Duration (min)")
        st.plotly_chart(fig, use_container_width=True)

        # Show reliability breakdown
        with st.expander("Reliability Score Breakdown"):
            breakdown = []
            breakdown.append(("Base Score", "+100", 100))
            running = 100
            if sim_distance > 20:
                breakdown.append(("Distance > 20 mi", "-40", -40)); running -= 40
            elif sim_distance > 10:
                breakdown.append(("Distance > 10 mi", "-25", -25)); running -= 25
            elif sim_distance > 5:
                breakdown.append(("Distance > 5 mi", "-10", -10)); running -= 10
            if sim_airport:
                breakdown.append(("Airport Trip", "-30", -30)); running -= 30
            if sim_rush:
                breakdown.append(("Rush Hour", "-10", -10)); running -= 10
            if 13 <= sim_hour <= 17:
                breakdown.append(("Hours 13–17", "-10", -10)); running -= 10
            if sim_cluster in [1, 3]:
                breakdown.append(("Cluster 1 or 3", "-15", -15)); running -= 15
            if sim_weekend:
                breakdown.append(("Weekend bonus", "+5", 5)); running += 5
            breakdown.append(("Final (clamped 0–100)", str(rel_score), 0))

            bd_df = pd.DataFrame(breakdown, columns=["Rule", "Points", "Delta"])
            st.dataframe(bd_df[["Rule", "Points"]], use_container_width=True, hide_index=True)

    # ── Tab 4: Outlier Trips ──────────────────────────────────────────────────
    with tab_outlier:
        st.markdown(seg_header("Outlier Trips — Prediction Failures"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Trips where the model prediction was significantly wrong. "
            "Adjust the error threshold to explore.</div>",
            unsafe_allow_html=True,
        )

        oc1, oc2 = st.columns(2)
        with oc1:
            err_mult = st.slider("Flag trips with |error| > N × MAE",
                                 1.0, 5.0, 2.0, 0.5, key="eta_outlier_mult")
        with oc2:
            rel_thresh = st.slider("Or reliability score <", 0, 100, 30, 5, key="eta_outlier_rel")

        outlier_threshold = mae * err_mult
        outlier_mask = (df["prediction_error_min"].abs() > outlier_threshold) | \
                       (df["reliability_score"] < rel_thresh)
        outliers = df[outlier_mask].copy()

        ok1, ok2, ok3 = st.columns(3)
        ok1.markdown(kpi_animated(f"{len(outliers)}", "Outlier Trips"), unsafe_allow_html=True)
        ok2.markdown(kpi_animated(f"{len(outliers)/len(df)*100:.1f}%", "Of Total"), unsafe_allow_html=True)
        if len(outliers) > 0:
            ok3.markdown(kpi_animated(f"{outliers['prediction_error_min'].abs().mean():.2f} min",
                                      "Avg Abs Error"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if len(outliers) > 0:
            show_cols = ["pickup_hour", "trip_distance", "lgbm_v3_pred_min",
                         "actual_duration_min", "prediction_error_min",
                         "reliability_score", "reliability_label", "distance_bin",
                         "rush_hour", "is_airport_pu"]
            st.dataframe(
                outliers[show_cols]
                .sort_values("prediction_error_min", key=abs, ascending=False)
                .head(100).reset_index(drop=True),
                use_container_width=True,
            )

            # Error distribution comparison
            st.markdown(seg_header("Outliers vs Normal — Error Comparison"), unsafe_allow_html=True)
            fig = go.Figure()
            normal = df[~outlier_mask]
            fig.add_trace(go.Histogram(x=normal["prediction_error_min"], nbinsx=40,
                                       name="Normal", marker_color=BLUE, opacity=0.6))
            fig.add_trace(go.Histogram(x=outliers["prediction_error_min"], nbinsx=20,
                                       name="Outliers", marker_color=RED, opacity=0.7))
            fig.update_layout(**DARK, height=280, barmode="overlay",
                              xaxis_title="Prediction Error (min)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            # Outlier characteristics
            st.markdown(seg_header("Outlier Characteristics"), unsafe_allow_html=True)
            oc_l, oc_r = st.columns(2)
            with oc_l:
                fig = go.Figure()
                for label, color in [("High", GREEN), ("Medium", YELLOW), ("Low", RED)]:
                    cnt_n = len(normal[normal["reliability_label"] == label])
                    cnt_o = len(outliers[outliers["reliability_label"] == label])
                    fig.add_trace(go.Bar(name=label, x=["Normal", "Outliers"],
                                        y=[cnt_n, cnt_o], marker_color=color))
                fig.update_layout(**DARK, height=260, barmode="stack",
                                  yaxis_title="Count", title="Reliability Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with oc_r:
                bins_present = [b for b in BIN_ORDER if b in outliers["distance_bin"].values]
                if bins_present:
                    out_by_bin = outliers["distance_bin"].value_counts().reindex(bins_present).fillna(0)
                    fig = go.Figure(go.Bar(
                        x=out_by_bin.index, y=out_by_bin.values,
                        marker_color=[BIN_COLORS.get(b, BLUE) for b in out_by_bin.index],
                    ))
                    fig.update_layout(**DARK, height=260, showlegend=False,
                                      xaxis_title="Distance Bin", yaxis_title="Outlier Count",
                                      title="Outliers by Distance Bin")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No outliers found with current thresholds.")

    # ── Tab 5: Trip Lookup ─────────────────────────────────────────────────────
    with tab_lookup:
        st.markdown(seg_header("Trip Lookup"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Filter trips by hour / reliability, then select an index to inspect all features.</div>",
            unsafe_allow_html=True,
        )

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            hour_f = st.selectbox("Filter by Hour", ["All"] + [str(h) for h in range(24)])
        with fc2:
            rel_f  = st.selectbox("Filter by Reliability", ["All", "High", "Medium", "Low"])
        with fc3:
            dist_f = st.selectbox("Filter by Distance Bin", ["All"] + BIN_ORDER)

        filt = df.copy()
        if hour_f != "All":
            filt = filt[filt["pickup_hour"] == int(hour_f)]
        if rel_f != "All":
            filt = filt[filt["reliability_label"] == rel_f]
        if dist_f != "All":
            filt = filt[filt["distance_bin"] == dist_f]

        if len(filt) == 0:
            st.warning("No trips match the selected filters.")
        else:
            trip_idx = st.number_input(
                f"Trip index (0 – {len(filt)-1})", min_value=0,
                max_value=len(filt) - 1, value=0, step=1,
            )
            row = filt.iloc[int(trip_idx)]

            # KPI cards
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.markdown(kpi(f"{row['lgbm_v3_pred_min']:.1f} min", "Predicted ETA"),
                         unsafe_allow_html=True)
            kc2.markdown(kpi(f"{row['actual_duration_min']:.1f} min", "Actual Duration"),
                         unsafe_allow_html=True)
            err_sign = "+" if row["prediction_error_min"] >= 0 else ""
            kc3.markdown(kpi(f"{err_sign}{row['prediction_error_min']:.2f} min", "Error"),
                         unsafe_allow_html=True)
            kc4.markdown(kpi(str(int(row["reliability_score"])), "Reliability Score"),
                         unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown("**Trip Details**")
                days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                details = {
                    "Pickup Time": str(row["tpep_pickup_datetime"]),
                    "Dropoff Time": str(row["tpep_dropoff_datetime"]),
                    "Pickup Hour": int(row["pickup_hour"]),
                    "Day of Week": days[int(row["day_of_week"]) % 7],
                    "Rush Hour": "Yes" if row["rush_hour"] else "No",
                    "Weekend": "Yes" if row["is_weekend"] else "No",
                    "Late Night": "Yes" if row["late_night"] else "No",
                    "Airport Pickup": "Yes" if row["is_airport_pu"] else "No",
                    "Airport Dropoff": "Yes" if row["is_airport_do"] else "No",
                    "Distance Bin": row["distance_bin"],
                }
                for k, v in details.items():
                    st.markdown(f"<span style='color:#6b7a99'>{k}:</span> **{v}**",
                                unsafe_allow_html=True)

            with ic2:
                st.markdown("**Spatial & Speed Features**")
                spatial = {
                    "Trip Distance": f"{row['trip_distance']:.3f} miles",
                    "Haversine Dist": f"{row['haversine_dist']:.3f} mi",
                    "Manhattan Dist": f"{row['manhattan_dist']:.3f} mi",
                    "Route Efficiency": f"{row['route_efficiency']:.4f}",
                    "Pickup Cluster": int(row["pickup_cluster"]),
                    "Speed 15m Ago": f"{row['prev_speed_15m']:.2f} mph",
                    "Speed 30m Ago": f"{row['prev_speed_30m']:.2f} mph",
                    "Speed 60m Ago": f"{row['prev_speed_60m']:.2f} mph",
                }
                for k, v in spatial.items():
                    st.markdown(f"<span style='color:#6b7a99'>{k}:</span> **{v}**",
                                unsafe_allow_html=True)

            rl = row["reliability_label"]
            st.markdown(
                f"<br>**Reliability Label:** {pill(rl)}",
                unsafe_allow_html=True,
            )

            # Show filtered table below
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(seg_header(f"Filtered Trips ({len(filt)} results)"), unsafe_allow_html=True)
            show_cols = ["tpep_pickup_datetime", "pickup_hour", "trip_distance",
                         "lgbm_v3_pred_min", "actual_duration_min",
                         "prediction_error_min", "reliability_label", "distance_bin",
                         "rush_hour", "is_airport_pu"]
            st.dataframe(filt[show_cols].reset_index(drop=True), use_container_width=True)

    # ── Tab 6: GenAI Q&A ──────────────────────────────────────────────────────
    with tab_genai:
        genai_panel(
            "eta", ETA_SYS,
            ["How accurate is the ETA model?",
             "When does the model underperform?",
             "Why is rush hour harder to predict?"],
        )
