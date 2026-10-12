"""
app.py — NYC Taxi Analytics Dashboard (Main Entry Point)
=========================================================
Run with:  streamlit run app.py

File structure expected:
  app.py
  data_loader.py
  styles.py
  genai.py
  seg_eda.py
  seg_eta.py
  seg_demand.py
  seg_path.py
  data/
    path_1000.csv                                   ← REQUIRED (bundled)
    path_20.csv                                     ← REQUIRED (bundled)
    taxi_2016_test_final_with_predictions.parquet   ← OPTIONAL (888 MB)
    final_results_demand_pred_2016.parquet          ← OPTIONAL (31.6 MB)
"""

import streamlit as st

st.set_page_config(
    page_title="NYC Taxi Analytics",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── imports after page config ─────────────────────────────────────────────────
from styles import apply_css, kpi
from data_loader import (
    load_path_1000, load_path_20,
    load_eta_parquet, load_demand_parquet,
    parquet_status,
)
import seg_eda, seg_eta, seg_demand, seg_path

apply_css()

# ── Hide sidebar + popover styling ────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hide the default Streamlit sidebar completely */
    [data-testid="stSidebar"]{display:none!important}
    [data-testid="collapsedControl"]{display:none!important}
    [data-testid="stSidebarCollapsedControl"]{display:none!important}

    /* Hamburger popover styling */
    [data-testid="stPopover"] > button{
        background:#13161d!important;
        border:1px solid #252b3b!important;
        border-radius:10px!important;
        color:#f5c518!important;
        font-size:1.4rem!important;
        padding:6px 12px!important;
        min-height:42px!important;
        min-width:42px!important;
        line-height:1!important;
        transition:all 0.2s ease!important;
    }
    [data-testid="stPopover"] > button:hover{
        background:#1c2030!important;
        border-color:#f5c518!important;
        box-shadow:0 0 12px rgba(245,197,24,0.15)!important;
    }
    [data-testid="stPopover"] > button:focus{
        box-shadow:0 0 0 2px rgba(245,197,24,0.3)!important;
    }
    /* Popover panel styling */
    [data-testid="stPopoverBody"]{
        background:#0d0f15!important;
        border:1px solid #252b3b!important;
        border-radius:12px!important;
        min-width:280px!important;
        box-shadow:0 8px 32px rgba(0,0,0,0.5)!important;
    }
    [data-testid="stPopoverBody"] *{
        color:#e8ecf4!important;
    }
    [data-testid="stPopoverBody"] hr{
        border-color:#252b3b!important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header Row: Hamburger (left) + Heading (center) ──────────────────────────
ham_col, heading_col, spacer_col = st.columns([1, 18, 1])

with heading_col:
    st.markdown(
        """
        <div style='text-align:center;padding:6px 0 24px'>
          <div style='font-family:Helvetica,Arial,sans-serif;font-size:2.4rem;color:#ffffff;font-weight:700;letter-spacing:-1px'>
            NYC Taxi Analytics Platform
          </div>
          <div style='font-family:Helvetica,Arial,sans-serif;font-size:0.95rem;color:#7eb8da;font-style:italic;letter-spacing:1.5px;margin-top:6px;text-transform:uppercase;font-weight:400'>
            End-to-End Urban Mobility Intelligence System
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with ham_col:
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    with st.popover("☰", use_container_width=False):
        # Logo / Brand
        st.markdown(
            """
            <div style='text-align:center;padding:8px 0'>
              <div style='font-family:Space Mono,monospace;font-size:1.3rem;color:#f5c518;font-weight:700'>NYC TAXI</div>
              <div style='font-size:.65rem;color:#6b7a99;letter-spacing:2px;text-transform:uppercase'>
                Analytics Platform
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # Navigation
        segment = st.radio(
            "Navigate",
            ["Data Overview & EDA",
             "ETA Prediction",
             "Demand Prediction",
             "Path Simulation"],
            label_visibility="collapsed",
        )

        st.divider()

        # File status panel
        status = parquet_status()
        st.markdown(
            "<div style='font-size:.72rem;color:#6b7a99;margin-bottom:6px'>"
            "<b style='color:#8b9dc3'>Data Files</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span class='file-ok'>✓ path_1000.csv</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span class='file-ok'>✓ path_20.csv</span>",
            unsafe_allow_html=True,
        )
        eta_badge = "file-ok" if status["eta"] else "file-missing"
        eta_icon  = "✓" if status["eta"] else "✗"
        dem_badge = "file-ok" if status["demand"] else "file-missing"
        dem_icon  = "✓" if status["demand"] else "✗"
        st.markdown(
            f"<span class='{eta_badge}'>{eta_icon} ETA parquet</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='{dem_badge}'>{dem_icon} Demand parquet</span>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            """
            <div style='font-size:.68rem;color:#6b7a99;line-height:1.7'>
            <b style='color:#8b9dc3'>Models</b><br>
            ETA: LightGBM v3 · MAE 2.306 min<br>
            Demand: PySpark GBT · R² 0.84<br>
            Path: Dijkstra (OSMnx)<br><br>
            <b style='color:#8b9dc3'>Dataset</b><br>
            NYC Yellow Taxi 2016 · 34.5M trips
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Load data ─────────────────────────────────────────────────────────────────
df_1000 = load_path_1000()
df_20   = load_path_20()

# Parquet loading is lazy — only triggered for the active segment
# ── Route ─────────────────────────────────────────────────────────────────────
if segment == "Data Overview & EDA":
    seg_eda.render(df_1000)

elif segment == "ETA Prediction":
    df_eta, is_full_eta = load_eta_parquet()
    seg_eta.render(df_eta, is_full_eta)

elif segment == "Demand Prediction":
    df_dem, is_full_dem = load_demand_parquet()
    seg_demand.render(df_dem, is_full_dem)

elif segment == "Path Simulation":
    seg_path.render(df_1000, df_20)
