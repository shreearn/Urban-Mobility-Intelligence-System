"""seg_path.py – Segment 4: Path Simulation"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles import (DARK, YELLOW, BLUE, GREEN, RED, ORANGE, MUTED,
                    BIN_COLORS, BIN_ORDER, seg_header, kpi, kpi_animated, pill, sim_card)
from genai import genai_panel, PATH_SYS


def render(df: pd.DataFrame, df20: pd.DataFrame):
    st.markdown("<div class='pg-title'>Path Simulation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pg-sub'>OSMnx · Dijkstra Shortest Path · NYC Road Network · 1,000 Trip Analysis</div>",
        unsafe_allow_html=True,
    )

    tab_viz, tab_compare, tab_live, tab_map, tab_table, tab_genai = st.tabs(
        ["Visualisations", "Route Comparison", "Live Path Finder",
         "Route Map", "20-Sample Table", "GenAI Q&A"]
    )

    bins_present = [b for b in BIN_ORDER if b in df["distance_bin"].values]

    # ── Tab 1: Visualisations ─────────────────────────────────────────────────
    with tab_viz:
        # KPIs — real data
        n_gt     = int(df["shortest_gt_actual"].sum())
        mean_diff= df["difference"].mean()
        pct_half = (df["difference"].abs() <= 0.5).mean() * 100
        avg_eff  = df["route_efficiency"].mean()

        c1, c2, c3, c4 = st.columns(4)
        for col, (v, l) in zip(
            [c1, c2, c3, c4],
            [
                (f"{n_gt}", "Trips: OSM > Actual"),
                (f"{mean_diff:+.3f} km", "Avg Difference"),
                (f"{pct_half:.1f}%", "Within 0.5 km Match"),
                (f"{avg_eff:.3f}", "Avg Route Efficiency"),
            ],
        ):
            col.markdown(kpi_animated(v, l), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Scatter + Boxplots ─────────────────────────────────────────
        cl, cr = st.columns(2)
        with cl:
            st.markdown(seg_header("Shortest vs Actual Distance — All 1,000 Trips"), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["actual_km"], y=df["shortest_km"], mode="markers",
                marker=dict(color=df["difference"], colorscale="RdYlGn_r",
                            size=5, opacity=0.7,
                            colorbar=dict(title=dict(text="Diff (km)", font=dict(color="#e8ecf4")),
                                          tickfont=dict(color="#e8ecf4"))),
                name="Trips",
                customdata=np.stack([df["distance_bin"], df["reliability_label"]], axis=-1),
                hovertemplate=(
                    "Actual: %{x:.2f} km<br>Shortest: %{y:.2f} km<br>"
                    "Bin: %{customdata[0]}<br>Reliability: %{customdata[1]}<extra></extra>"
                ),
            ))
            mx = max(df["actual_km"].max(), df["shortest_km"].max())
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                                     line=dict(color=YELLOW, dash="dash", width=2),
                                     name="1:1 Line"))
            fig.update_layout(**DARK, height=360,
                              xaxis_title="Actual Distance (km)",
                              yaxis_title="Shortest Path Distance (km)")
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown(seg_header("Difference Distribution by Distance Bin"), unsafe_allow_html=True)
            fig = go.Figure()
            for b in bins_present:
                sub = df[df["distance_bin"] == b]["difference"]
                fig.add_trace(go.Box(y=sub, name=b, marker_color=BIN_COLORS[b],
                                     boxmean=True))
            fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.4)
            fig.update_layout(**DARK, height=360, showlegend=False,
                              yaxis_title="Difference: Shortest – Actual (km)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Route efficiency + OSM>actual rate ─────────────────────────
        cl2, cr2 = st.columns(2)
        with cl2:
            st.markdown(seg_header("Route Efficiency by Distance Bin"), unsafe_allow_html=True)
            eff = df.groupby("distance_bin")["route_efficiency"].mean().reindex(bins_present)
            fig = go.Figure(go.Bar(
                x=eff.index, y=eff.values,
                marker_color=[BIN_COLORS[b] for b in eff.index],
                text=[f"{v:.3f}" for v in eff.values], textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.add_hline(y=1.0, line_color=YELLOW, line_dash="dash",
                          annotation_text="Ideal = 1.0 (perfectly direct)",
                          annotation_font_color=YELLOW)
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Distance Bin", yaxis_title="Avg Route Efficiency")
            st.plotly_chart(fig, use_container_width=True)

        with cr2:
            st.markdown(seg_header("OSM Shortest > Actual Rate by Bin"), unsafe_allow_html=True)
            sgt = df.groupby("distance_bin")["shortest_gt_actual"].mean().reindex(bins_present) * 100
            fig = go.Figure(go.Bar(
                x=sgt.index, y=sgt.values,
                marker_color=[BIN_COLORS[b] for b in sgt.index],
                text=[f"{v:.1f}%" for v in sgt.values], textposition="outside",
                textfont=dict(color="#e8ecf4"),
            ))
            fig.update_layout(**DARK, height=300, showlegend=False,
                              xaxis_title="Distance Bin",
                              yaxis_title="% Trips where OSM route > Actual")
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: Cluster route efficiency ───────────────────────────────────
        st.markdown(seg_header("Route Efficiency by Pickup Cluster"), unsafe_allow_html=True)
        cl_eff = df.groupby("pickup_cluster")["route_efficiency"].mean().reset_index()
        cl_eff = cl_eff.sort_values("route_efficiency", ascending=False)
        fig = go.Figure(go.Bar(
            x=cl_eff["pickup_cluster"].astype(str), y=cl_eff["route_efficiency"],
            marker=dict(color=cl_eff["route_efficiency"],
                        colorscale=[[0, GREEN],[0.5, YELLOW],[1, RED]]),
            text=[f"{v:.3f}" for v in cl_eff["route_efficiency"]],
            textposition="outside", textfont=dict(color="#e8ecf4"),
        ))
        fig.add_hline(y=df["route_efficiency"].mean(), line_color="white",
                      line_dash="dot", line_width=1, opacity=0.5)
        fig.update_layout(**DARK, height=280, showlegend=False,
                          xaxis_title="Pickup Cluster",
                          yaxis_title="Avg Route Efficiency")
        st.plotly_chart(fig, use_container_width=True)

        # ── Row 4: Prediction error vs route efficiency ───────────────────────
        st.markdown(seg_header("Prediction Error vs Route Efficiency (Scatter)"), unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(
            x=df["route_efficiency"].clip(upper=5), y=df["prediction_error_min"].abs(),
            mode="markers",
            marker=dict(color=df["reliability_score"], colorscale="RdYlGn",
                        size=5, opacity=0.55,
                        colorbar=dict(title=dict(text="Reliability", font=dict(color="#e8ecf4")),
                                      tickfont=dict(color="#e8ecf4"))),
            customdata=df["distance_bin"],
            hovertemplate="Efficiency: %{x:.3f}<br>Abs Error: %{y:.2f} min<br>Bin: %{customdata}<extra></extra>",
        ))
        fig.update_layout(**DARK, height=280, showlegend=False,
                          xaxis_title="Route Efficiency (capped at 5)",
                          yaxis_title="Absolute Prediction Error (min)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Route Comparison ───────────────────────────────────────────────
    with tab_compare:
        st.markdown(seg_header("Side-by-Side Route Comparison"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Select two trips to compare their routes, distances, and prediction accuracy.</div>",
            unsafe_allow_html=True,
        )

        # Filter
        cmp_bin = st.selectbox("Filter by Distance Bin", ["All"] + BIN_ORDER, key="cmp_bin")
        cmp_df = df.copy()
        if cmp_bin != "All":
            cmp_df = cmp_df[cmp_df["distance_bin"] == cmp_bin]

        if len(cmp_df) < 2:
            st.warning("Not enough trips in selected bin for comparison.")
        else:
            cc1, cc2 = st.columns(2)
            with cc1:
                trip_a = st.number_input("Trip A index", 0, len(cmp_df)-1, 0, key="cmp_a")
            with cc2:
                trip_b = st.number_input("Trip B index", 0, len(cmp_df)-1,
                                         min(1, len(cmp_df)-1), key="cmp_b")

            row_a = cmp_df.iloc[int(trip_a)]
            row_b = cmp_df.iloc[int(trip_b)]

            # Side-by-side KPIs
            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_mid, col_b = st.columns([5, 1, 5])

            with col_a:
                st.markdown(
                    f"<div style='text-align:center;font-family:Space Mono;font-size:1rem;"
                    f"color:{BLUE};margin-bottom:8px'>🅰 Trip A</div>",
                    unsafe_allow_html=True,
                )
                ak1, ak2 = st.columns(2)
                ak1.markdown(kpi(f"{row_a['actual_km']:.2f} km", "Actual Dist"), unsafe_allow_html=True)
                ak2.markdown(kpi(f"{row_a['shortest_km']:.2f} km", "Shortest Path"), unsafe_allow_html=True)
                ak3, ak4 = st.columns(2)
                ak3.markdown(kpi(f"{row_a['difference']:+.3f} km", "Difference"), unsafe_allow_html=True)
                ak4.markdown(kpi(f"{row_a['route_efficiency']:.3f}", "Efficiency"), unsafe_allow_html=True)
                ak5, ak6 = st.columns(2)
                ak5.markdown(kpi(f"{row_a['lgbm_v3_pred_min']:.1f} min", "Pred ETA"), unsafe_allow_html=True)
                ak6.markdown(kpi(f"{row_a['actual_duration_min']:.1f} min", "Actual ETA"), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='text-align:center;margin:6px 0'>"
                    f"{pill(row_a['reliability_label'])} &nbsp; "
                    f"<span style='color:#6b7a99;font-size:.8rem'>{row_a['distance_bin']}</span></div>",
                    unsafe_allow_html=True,
                )

            with col_mid:
                st.markdown(
                    "<div style='text-align:center;font-size:2rem;color:#252b3b;margin-top:80px'>⟷</div>",
                    unsafe_allow_html=True,
                )

            with col_b:
                st.markdown(
                    f"<div style='text-align:center;font-family:Space Mono;font-size:1rem;"
                    f"color:{YELLOW};margin-bottom:8px'>🅱 Trip B</div>",
                    unsafe_allow_html=True,
                )
                bk1, bk2 = st.columns(2)
                bk1.markdown(kpi(f"{row_b['actual_km']:.2f} km", "Actual Dist"), unsafe_allow_html=True)
                bk2.markdown(kpi(f"{row_b['shortest_km']:.2f} km", "Shortest Path"), unsafe_allow_html=True)
                bk3, bk4 = st.columns(2)
                bk3.markdown(kpi(f"{row_b['difference']:+.3f} km", "Difference"), unsafe_allow_html=True)
                bk4.markdown(kpi(f"{row_b['route_efficiency']:.3f}", "Efficiency"), unsafe_allow_html=True)
                bk5, bk6 = st.columns(2)
                bk5.markdown(kpi(f"{row_b['lgbm_v3_pred_min']:.1f} min", "Pred ETA"), unsafe_allow_html=True)
                bk6.markdown(kpi(f"{row_b['actual_duration_min']:.1f} min", "Actual ETA"), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='text-align:center;margin:6px 0'>"
                    f"{pill(row_b['reliability_label'])} &nbsp; "
                    f"<span style='color:#6b7a99;font-size:.8rem'>{row_b['distance_bin']}</span></div>",
                    unsafe_allow_html=True,
                )

            # Comparison chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(seg_header("Metric Comparison"), unsafe_allow_html=True)
            metrics = ["Actual (km)", "Shortest (km)", "Pred ETA (min)", "Actual ETA (min)", "Efficiency"]
            vals_a = [row_a["actual_km"], row_a["shortest_km"], row_a["lgbm_v3_pred_min"],
                      row_a["actual_duration_min"], row_a["route_efficiency"]]
            vals_b = [row_b["actual_km"], row_b["shortest_km"], row_b["lgbm_v3_pred_min"],
                      row_b["actual_duration_min"], row_b["route_efficiency"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Trip A", x=metrics, y=vals_a, marker_color=BLUE,
                                 text=[f"{v:.2f}" for v in vals_a], textposition="outside",
                                 textfont=dict(color="#e8ecf4")))
            fig.add_trace(go.Bar(name="Trip B", x=metrics, y=vals_b, marker_color=YELLOW,
                                 text=[f"{v:.2f}" for v in vals_b], textposition="outside",
                                 textfont=dict(color="#e8ecf4")))
            fig.update_layout(**DARK, height=320, barmode="group")
            st.plotly_chart(fig, use_container_width=True)

            # Map comparison
            try:
                import folium
                from streamlit_folium import st_folium

                st.markdown(seg_header("Map Comparison"), unsafe_allow_html=True)
                mc1, mc2 = st.columns(2)

                for col, row, label, color in [(mc1, row_a, "Trip A", BLUE),
                                                (mc2, row_b, "Trip B", YELLOW)]:
                    with col:
                        center_lat = (row["pickup_latitude"] + row["dropoff_latitude"]) / 2
                        center_lon = (row["pickup_longitude"] + row["dropoff_longitude"]) / 2
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                                       tiles="CartoDB dark_matter")
                        folium.CircleMarker(
                            [row["pickup_latitude"], row["pickup_longitude"]],
                            radius=10, color=color, fill=True,
                            fill_color=color, fill_opacity=0.9,
                            tooltip=f"{label} Pickup",
                        ).add_to(m)
                        folium.CircleMarker(
                            [row["dropoff_latitude"], row["dropoff_longitude"]],
                            radius=7, color=color, fill=True,
                            fill_color="#0a0c10", fill_opacity=0.85,
                            tooltip=f"{label} Dropoff",
                        ).add_to(m)
                        folium.PolyLine(
                            [[row["pickup_latitude"], row["pickup_longitude"]],
                             [row["dropoff_latitude"], row["dropoff_longitude"]]],
                            color=color, weight=2, opacity=0.65, dash_array="6",
                        ).add_to(m)
                        st_folium(m, width=None, height=350, returned_objects=[],
                                  key=f"cmp_map_{label}")
            except ImportError:
                st.info("Install `folium` and `streamlit-folium` for map comparison.")

    # ── Tab 3: Live Path Finder ───────────────────────────────────────────────
    with tab_live:
        st.markdown(seg_header("Real-Time Path Simulation"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:14px'>"
            "Compute the shortest driving path between any two points in Manhattan "
            "using the OSMnx road network and Dijkstra's algorithm.</div>",
            unsafe_allow_html=True,
        )

        try:
            from path_engine import load_graph, compute_shortest_path, LANDMARKS

            # Input method
            input_method = st.radio(
                "Input method",
                ["Pick from landmarks", "Pick from trip data", "Enter coordinates"],
                horizontal=True, key="path_input_method",
            )

            landmark_names = list(LANDMARKS.keys())

            if input_method == "Pick from landmarks":
                pc1, pc2 = st.columns(2)
                with pc1:
                    orig_name = st.selectbox("Pickup Location", landmark_names, index=0, key="path_orig")
                    orig_lat, orig_lon = LANDMARKS[orig_name]
                    st.caption(f"{orig_lat:.4f}, {orig_lon:.4f}")
                with pc2:
                    dest_name = st.selectbox("Dropoff Location", landmark_names, index=6, key="path_dest")
                    dest_lat, dest_lon = LANDMARKS[dest_name]
                    st.caption(f"{dest_lat:.4f}, {dest_lon:.4f}")

            elif input_method == "Pick from trip data":
                trip_idx = st.number_input("Trip index (0–999)", 0, len(df)-1, 0, key="path_trip_idx")
                trip_row = df.iloc[int(trip_idx)]
                orig_lat = trip_row["pickup_latitude"]
                orig_lon = trip_row["pickup_longitude"]
                dest_lat = trip_row["dropoff_latitude"]
                dest_lon = trip_row["dropoff_longitude"]
                st.markdown(
                    f"<div style='color:#6b7a99;font-size:.82rem'>"
                    f"Pickup: ({orig_lat:.4f}, {orig_lon:.4f}) → "
                    f"Dropoff: ({dest_lat:.4f}, {dest_lon:.4f}) | "
                    f"Actual: {trip_row['actual_km']:.2f} km | "
                    f"Bin: {trip_row['distance_bin']}</div>",
                    unsafe_allow_html=True,
                )

            else:  # Manual coordinates
                pc1, pc2 = st.columns(2)
                with pc1:
                    orig_lat = st.number_input("Pickup Latitude", value=40.7580, format="%.6f", key="path_olat")
                    orig_lon = st.number_input("Pickup Longitude", value=-73.9855, format="%.6f", key="path_olon")
                with pc2:
                    dest_lat = st.number_input("Dropoff Latitude", value=40.7074, format="%.6f", key="path_dlat")
                    dest_lon = st.number_input("Dropoff Longitude", value=-74.0113, format="%.6f", key="path_dlon")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Calculate Shortest Path", key="path_calc", type="primary"):
                with st.spinner("Loading road network + computing Dijkstra shortest path…"):
                    G = load_graph()
                    result = compute_shortest_path(G, orig_lat, orig_lon, dest_lat, dest_lon)

                if result["error"]:
                    st.error(f"Path computation error: {result['error']}")
                else:
                    # Results KPIs
                    rk1, rk2, rk3, rk4 = st.columns(4)
                    rk1.markdown(sim_card(f"{result['distance_km']:.2f} km", "Shortest Path",
                                          GREEN), unsafe_allow_html=True)
                    rk2.markdown(sim_card(f"{result['haversine_km']:.2f} km", "Straight Line",
                                          BLUE), unsafe_allow_html=True)
                    eff = result['route_efficiency']
                    eff_color = GREEN if eff < 1.3 else YELLOW if eff < 1.8 else RED
                    rk3.markdown(sim_card(f"{eff:.3f}", "Route Efficiency", eff_color,
                                          "1.0 = perfectly direct"), unsafe_allow_html=True)
                    rk4.markdown(sim_card(f"{result['num_nodes']}", "Road Nodes", MUTED,
                                          "Intersections traversed"), unsafe_allow_html=True)

                    # If trip data input, show comparison
                    if input_method == "Pick from trip data":
                        st.markdown(seg_header("Comparison with Actual Trip Data"), unsafe_allow_html=True)
                        cmp1, cmp2, cmp3 = st.columns(3)
                        cmp1.markdown(kpi(f"{trip_row['actual_km']:.2f} km", "Actual (Taxi Meter)"),
                                      unsafe_allow_html=True)
                        cmp2.markdown(kpi(f"{result['distance_km']:.2f} km", "OSM Shortest"),
                                      unsafe_allow_html=True)
                        diff = result['distance_km'] - trip_row['actual_km']
                        diff_color = RED if diff > 0 else GREEN
                        cmp3.markdown(kpi(f"{diff:+.2f} km", "Difference"),
                                      unsafe_allow_html=True)

                    # Map with route
                    st.markdown(seg_header("Route Visualization"), unsafe_allow_html=True)
                    try:
                        import folium
                        from streamlit_folium import st_folium

                        center_lat = (orig_lat + dest_lat) / 2
                        center_lon = (orig_lon + dest_lon) / 2
                        m = folium.Map(location=[center_lat, center_lon],
                                       zoom_start=13, tiles="CartoDB dark_matter")

                        # Draw animated route
                        if result["path_coords"]:
                            from folium.plugins import AntPath
                            AntPath(
                                result["path_coords"],
                                color=GREEN, weight=5, opacity=0.85,
                                delay=800, dash_array=[20, 40],
                                pulse_color="#ffffff",
                                tooltip=f"Shortest path: {result['distance_km']:.2f} km",
                            ).add_to(m)

                        # Straight-line dashed
                        folium.PolyLine(
                            [[orig_lat, orig_lon], [dest_lat, dest_lon]],
                            color=RED, weight=2, opacity=0.6, dash_array="8",
                            tooltip=f"Haversine: {result['haversine_km']:.2f} km",
                        ).add_to(m)

                        # Markers
                        folium.CircleMarker(
                            [orig_lat, orig_lon], radius=12,
                            color=YELLOW, fill=True, fill_color=YELLOW, fill_opacity=0.9,
                            tooltip="Pickup",
                        ).add_to(m)
                        folium.CircleMarker(
                            [dest_lat, dest_lon], radius=12,
                            color=RED, fill=True, fill_color=RED, fill_opacity=0.9,
                            tooltip="Dropoff",
                        ).add_to(m)

                        # Legend
                        legend = """
                        <div style='position:fixed;bottom:28px;left:28px;background:#13161d;
                                    border:1px solid #252b3b;padding:12px 16px;border-radius:8px;
                                    font-family:monospace;font-size:12px;color:#e8ecf4;z-index:9999'>
                        <b style='color:#f5c518'>Route Legend</b><br>
                        <span style='color:#22c55e'>━ ━ ━</span> Animated shortest path<br>
                        <span style='color:#ef4444'>╌╌</span> Straight line (haversine)<br>
                        <span style='color:#f5c518'>●</span> Pickup<br>
                        <span style='color:#ef4444'>●</span> Dropoff
                        </div>
                        """
                        m.get_root().html.add_child(folium.Element(legend))

                        st_folium(m, width=None, height=500, returned_objects=[],
                                  key="live_path_map")

                    except ImportError:
                        st.warning("Install `folium` and `streamlit-folium` for map display.")

        except ImportError:
            st.error(
                "**OSMnx not installed.** Run `pip install osmnx networkx` to enable "
                "real-time path simulation."
            )
            st.info("The Live Path Finder requires the OSMnx library to download and query "
                    "the NYC road network graph.")

    # ── Tab 4: Route Map ──────────────────────────────────────────────────────
    with tab_map:
        st.markdown(seg_header("Interactive Route Map — 20 Sample Trips"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:12px'>"
            "Click any marker for trip metrics. Dashed lines connect pickup → dropoff "
            "(actual road polylines require loading the NYC graph at runtime). "
            "Colour = distance bin.</div>",
            unsafe_allow_html=True,
        )

        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(location=[40.745, -73.97], zoom_start=11,
                           tiles="CartoDB dark_matter")

            for _, row in df20.iterrows():
                color = BIN_COLORS.get(str(row["distance_bin"]), YELLOW)

                popup_html = f"""
                <div style='font-family:monospace;font-size:12px;min-width:180px'>
                <b style='color:{color}'>{row['distance_bin']}</b><br>
                Actual: {row['actual_km']:.3f} km<br>
                Shortest: {row['shortest_km']:.3f} km<br>
                Difference: {row['difference']:+.3f} km<br>
                Pred ETA: {row['lgbm_v3_pred_min']:.1f} min<br>
                Actual ETA: {row['actual_duration_min']:.1f} min<br>
                Error: {row['prediction_error_min']:+.2f} min<br>
                Reliability: {row['reliability_label']}
                </div>
                """

                # Pickup
                folium.CircleMarker(
                    location=[row["pickup_latitude"], row["pickup_longitude"]],
                    radius=9, color=color, fill=True,
                    fill_color=color, fill_opacity=0.9,
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=f"Pickup · {row['distance_bin']}",
                ).add_to(m)

                # Dropoff
                folium.CircleMarker(
                    location=[row["dropoff_latitude"], row["dropoff_longitude"]],
                    radius=6, color=color, fill=True,
                    fill_color="#0a0c10", fill_opacity=0.85,
                    tooltip=f"Dropoff · {row['distance_bin']}",
                ).add_to(m)

                # Dashed line
                folium.PolyLine(
                    locations=[
                        [row["pickup_latitude"],  row["pickup_longitude"]],
                        [row["dropoff_latitude"], row["dropoff_longitude"]],
                    ],
                    color=color, weight=2, opacity=0.65, dash_array="6",
                ).add_to(m)

            # Legend
            legend = """
            <div style='position:fixed;bottom:28px;left:28px;background:#13161d;
                        border:1px solid #252b3b;padding:12px 16px;border-radius:8px;
                        font-family:monospace;font-size:12px;color:#e8ecf4;z-index:9999'>
            <b style='color:#f5c518'>Distance Bins</b><br>
            <span style='color:#22c55e'>●</span> Short (0–2 km)<br>
            <span style='color:#3b82f6'>●</span> Medium (2–5 km)<br>
            <span style='color:#f5c518'>●</span> Long (5–10 km)<br>
            <span style='color:#ef4444'>●</span> Very Long (10+ km)<br>
            <span style='color:#888'>○</span> Dropoff point
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend))

            st_folium(m, width=None, height=540, returned_objects=[])

        except ImportError:
            st.warning("Install `folium` and `streamlit-folium` to see the interactive map.")
            # Plotly fallback
            bin_colors_list = [BIN_COLORS.get(str(b), YELLOW) for b in df20["distance_bin"]]
            fig = go.Figure()
            fig.add_trace(go.Scattermapbox(
                lat=df20["pickup_latitude"], lon=df20["pickup_longitude"],
                mode="markers", marker=dict(size=12, color=bin_colors_list),
                name="Pickup",
                text=df20["distance_bin"],
            ))
            fig.add_trace(go.Scattermapbox(
                lat=df20["dropoff_latitude"], lon=df20["dropoff_longitude"],
                mode="markers", marker=dict(size=8, color=bin_colors_list, symbol="circle-open"),
                name="Dropoff",
            ))
            fig.update_layout(
                mapbox=dict(style="carto-darkmatter", center=dict(lat=40.745, lon=-73.97), zoom=10),
                **{k: v for k, v in DARK.items() if k not in ["xaxis","yaxis"]},
                height=500, margin=dict(t=0, b=0, l=0, r=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: 20-Sample Table ────────────────────────────────────────────────
    with tab_table:
        st.markdown(seg_header("20-Sample Route Analysis — Full Detail"), unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#6b7a99;font-size:.83rem;margin-bottom:12px'>"
            "5 trips selected per distance bin (Short / Medium / Long / Very Long).</div>",
            unsafe_allow_html=True,
        )

        show_cols_20 = [
            "tpep_pickup_datetime", "distance_bin",
            "actual_km", "shortest_km", "difference", "shortest_gt_actual",
            "route_efficiency", "lgbm_v3_pred_min", "actual_duration_min",
            "prediction_error_min", "reliability_label", "reliability_score",
            "pickup_hour", "rush_hour", "is_airport_pu", "is_airport_do",
        ]
        display_20 = df20[show_cols_20].copy()
        display_20.columns = [
            "Pickup Time", "Bin",
            "Actual (km)", "Shortest (km)", "Diff (km)", "OSM>Actual?",
            "Efficiency", "Pred ETA (min)", "Actual ETA (min)",
            "Error (min)", "Reliability", "Rel Score",
            "Hour", "Rush?", "Airport PU", "Airport DO",
        ]
        st.dataframe(display_20, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(seg_header("Bin-by-Bin Summary (Real Data)"), unsafe_allow_html=True)
        summary = df.groupby("distance_bin").agg(
            Count=("actual_km","count"),
            Avg_Actual_km=("actual_km","mean"),
            Avg_Shortest_km=("shortest_km","mean"),
            Avg_Difference=("difference","mean"),
            Pct_OSM_GT_Actual=("shortest_gt_actual","mean"),
            Avg_Route_Efficiency=("route_efficiency","mean"),
            Avg_MAE_min=("prediction_error_min", lambda x: x.abs().mean()),
        ).reindex(bins_present).round(3)
        summary["Pct_OSM_GT_Actual"] = (summary["Pct_OSM_GT_Actual"] * 100).round(1).astype(str) + "%"
        st.dataframe(summary, use_container_width=True)

    # ── Tab 6: GenAI Q&A ──────────────────────────────────────────────────────
    with tab_genai:
        genai_panel(
            "path", PATH_SYS,
            ["What algorithm was used for shortest path?",
             "Is the shortest path always the fastest?",
             "Why does OSM sometimes give a longer route than actual?"],
        )
