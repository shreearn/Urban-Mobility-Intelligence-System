"""styles.py – Shared CSS, colours, and Plotly layout for the dashboard."""

import plotly.graph_objects as go
import re, uuid

# ── Brand colours ─────────────────────────────────────────────────────────────
YELLOW   = "#f5c518"
ORANGE   = "#e8521a"
GREEN    = "#22c55e"
BLUE     = "#3b82f6"
RED      = "#ef4444"
PURPLE   = "#a855f7"
TEAL     = "#14b8a6"

BG       = "#0a0c10"
SURFACE  = "#13161d"
SURFACE2 = "#1c2030"
BORDER   = "#252b3b"
TEXT     = "#e8ecf4"
MUTED    = "#6b7a99"

# ── Plotly dark layout base ───────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor="#0f1118",
    font=dict(family="Helvetica, Arial, sans-serif", color=TEXT, size=12),
    xaxis=dict(gridcolor=SURFACE2, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=SURFACE2, linecolor=BORDER, zerolinecolor=BORDER),
    margin=dict(t=44, b=44, l=54, r=24),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
)

BIN_COLORS = {
    "Short (0-2 km)":     GREEN,
    "Medium (2-5 km)":    BLUE,
    "Long (5-10 km)":     YELLOW,
    "Very Long (10+ km)": RED,
}
BIN_ORDER = ["Short (0-2 km)", "Medium (2-5 km)", "Long (5-10 km)", "Very Long (10+ km)"]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0a0c10!important;color:#e8ecf4!important;font-family:Helvetica,Arial,sans-serif}

/* ── Sidebar styling ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"]{background:#0d0f15!important;border-right:1px solid #252b3b}
[data-testid="stSidebar"] *{color:#e8ecf4}

/* Collapse button styling (hamburger) */
[data-testid="stSidebar"] button[kind="header"]{
    color:#f5c518!important;
}
[data-testid="collapsedControl"]{
    color:#f5c518!important;
}
/* Style the sidebar collapse/expand arrow */
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button{
    background:#13161d!important;
    border:1px solid #252b3b!important;
    border-radius:8px!important;
    color:#f5c518!important;
}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover{
    background:#1c2030!important;
    border-color:#f5c518!important;
}
/* When sidebar is collapsed, style the expand button */
[data-testid="collapsedControl"] button{
    background:#13161d!important;
    border:1px solid #252b3b!important;
    border-radius:8px!important;
    color:#f5c518!important;
    box-shadow:0 2px 8px rgba(0,0,0,0.3)!important;
}
[data-testid="collapsedControl"] button:hover{
    background:#1c2030!important;
    border-color:#f5c518!important;
    box-shadow:0 2px 12px rgba(245,197,24,0.15)!important;
}
/* Sidebar transition for smooth open/close */
[data-testid="stSidebar"] > div:first-child{
    transition: width 0.3s ease, transform 0.3s ease;
}

.block-container{padding-top:2.8rem!important}

/* hide Streamlit deploy / header bar background but keep controls visible */
header[data-testid="stHeader"]{background:transparent!important;pointer-events:none}
header[data-testid="stHeader"] *{pointer-events:auto}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}

/* headings */
.pg-title{font-family:Helvetica,Arial,sans-serif;font-size:2.2rem;font-weight:700;color:#ffffff;line-height:1.2;margin-bottom:4px;padding-top:4px}
.pg-sub{font-size:.85rem;color:#7eb8da;font-style:italic;margin-bottom:1.6rem}
.seg-hdr{font-family:Helvetica,Arial,sans-serif;font-size:1.05rem;font-weight:700;color:#e8ecf4;border-left:3px solid #f5c518;padding-left:10px;margin:18px 0 10px}

/* metric card */
.kpi-card{background:#13161d;border:1px solid #252b3b;border-radius:10px;padding:16px 18px;text-align:center;margin-bottom:6px}
.kpi-val{font-family:Helvetica,Arial,sans-serif;font-size:1.75rem;font-weight:700;color:#f5c518}
.kpi-lbl{font-size:.72rem;color:#6b7a99;text-transform:uppercase;letter-spacing:.8px;margin-top:3px}

/* CSS-only animated KPI — fade-in + slide-up effect */
@keyframes kpiFadeIn{
    0%{opacity:0;transform:translateY(12px)}
    100%{opacity:1;transform:translateY(0)}
}
.kpi-anim{
    font-family:Helvetica,Arial,sans-serif;font-size:1.75rem;font-weight:700;color:#f5c518;
    display:inline-block;
    animation:kpiFadeIn 0.6s ease-out both;
}

/* Staggered animation delays for KPI cards in a row */
.kpi-card:nth-child(1) .kpi-anim{animation-delay:0.05s}
.kpi-card:nth-child(2) .kpi-anim{animation-delay:0.12s}
.kpi-card:nth-child(3) .kpi-anim{animation-delay:0.20s}
.kpi-card:nth-child(4) .kpi-anim{animation-delay:0.28s}
.kpi-card:nth-child(5) .kpi-anim{animation-delay:0.36s}

/* data pills */
.pill-high{background:#14532d;color:#86efac;padding:2px 10px;border-radius:20px;font-size:.75rem;font-family:Helvetica,Arial,sans-serif}
.pill-med{background:#713f12;color:#fbbf24;padding:2px 10px;border-radius:20px;font-size:.75rem;font-family:Helvetica,Arial,sans-serif}
.pill-low{background:#7f1d1d;color:#fca5a5;padding:2px 10px;border-radius:20px;font-size:.75rem;font-family:Helvetica,Arial,sans-serif}

/* chat */
.chat-user{background:#1c2030;border:1px solid #252b3b;border-radius:12px 12px 4px 12px;padding:12px 16px;margin:8px 0;font-size:.88rem;text-align:right}
.chat-ai{background:linear-gradient(135deg,#141c3a,#1c2030);border:1px solid #2a3560;border-radius:12px 12px 12px 4px;padding:14px 18px;margin:8px 0;font-size:.88rem;line-height:1.65}
.chat-lbl{font-family:Helvetica,Arial,sans-serif;font-size:.65rem;color:#e8ecf4;margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px}

/* file status badge */
.file-ok{background:#14532d;color:#86efac;padding:4px 12px;border-radius:6px;font-size:.78rem;display:inline-block;margin:2px}
.file-missing{background:#7f1d1d;color:#fca5a5;padding:4px 12px;border-radius:6px;font-size:.78rem;display:inline-block;margin:2px}
.file-info{background:#1e3a5f;color:#93c5fd;padding:4px 12px;border-radius:6px;font-size:.78rem;display:inline-block;margin:2px}

/* tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:#13161d!important;border-bottom:1px solid #252b3b;gap:8px;padding:0 4px}
[data-testid="stTabs"] [data-baseweb="tab"]{font-family:Helvetica,Arial,sans-serif;font-size:.78rem;color:#6b7a99!important;background:transparent!important;border:none;padding:10px 16px!important;letter-spacing:.3px}
[data-testid="stTabs"] [aria-selected="true"]{color:#ffffff!important;border-bottom:2px solid #f5c518!important}

/* inputs */
.stTextArea textarea,.stSelectbox>div,.stNumberInput input,.stTextInput input{background:#1c2030!important;border:1px solid #252b3b!important;color:#e8ecf4!important;border-radius:8px}
.stButton>button{background:#f5c518!important;color:#0a0c10!important;font-family:Helvetica,Arial,sans-serif;font-weight:700;border:none!important;border-radius:8px}
.stButton>button:hover{background:#e6b800!important}

/* expanders & dividers */
div[data-testid="stExpander"]{background:#13161d!important;border:1px solid #252b3b!important;border-radius:8px}
hr{border-color:#252b3b}

/* dataframe */
[data-testid="stDataFrame"]{border:1px solid #252b3b;border-radius:8px}

/* gauge / simulator cards */
.sim-card{background:linear-gradient(135deg,#13161d,#1a1f2e);border:1px solid #252b3b;border-radius:14px;padding:24px 28px;text-align:center;margin:8px 0}
.sim-val{font-family:Helvetica,Arial,sans-serif;font-size:2.8rem;font-weight:700;line-height:1.1}
.sim-lbl{font-size:.78rem;color:#6b7a99;text-transform:uppercase;letter-spacing:1px;margin-top:6px}
.sim-sub{font-size:.82rem;color:#8b9dc3;margin-top:4px}

/* anomaly row highlight */
.anomaly-badge{background:#7f1d1d;color:#fca5a5;padding:2px 8px;border-radius:4px;font-size:.72rem;font-family:Helvetica,Arial,sans-serif}

/* ── Sidebar nav items styling ─────────────────────────────────────────────── */
.sidebar-nav{padding:4px 0}
.sidebar-nav-item{
    display:block;
    padding:10px 16px;
    margin:2px 0;
    border-radius:8px;
    color:#8b9dc3;
    font-family:Helvetica,Arial,sans-serif;
    font-size:.88rem;
    font-weight:500;
    text-decoration:none;
    transition:all 0.2s ease;
    cursor:pointer;
}
.sidebar-nav-item:hover{
    background:#1c2030;
    color:#e8ecf4;
}
.sidebar-nav-item.active{
    background:linear-gradient(135deg,#1a1f2e,#1c2030);
    color:#f5c518;
    border-left:3px solid #f5c518;
    font-weight:600;
}

/* ── Sidebar info cards ────────────────────────────────────────────────────── */
.sidebar-info{
    background:#13161d;
    border:1px solid #1c2030;
    border-radius:8px;
    padding:12px 14px;
    margin:8px 0;
}
</style>
"""


def apply_css():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(val: str, label: str) -> str:
    return f"<div class='kpi-card'><div class='kpi-val'>{val}</div><div class='kpi-lbl'>{label}</div></div>"


def kpi_animated(val: str, label: str) -> str:
    """Animated KPI card — uses CSS fade-in animation instead of broken JS counters.
    Simply displays the value with a smooth entrance animation."""
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-anim'>{val}</div>"
        f"<div class='kpi-lbl'>{label}</div>"
        f"</div>"
    )


def seg_header(text: str) -> str:
    return f"<div class='seg-hdr'>{text}</div>"


def pill(label: str) -> str:
    cls = {"High": "pill-high", "Medium": "pill-med", "Low": "pill-low"}.get(label, "pill-med")
    return f"<span class='{cls}'>{label}</span>"


def sim_card(val: str, label: str, color: str = YELLOW, sub: str = "") -> str:
    """Big simulation result card used by What-If and Forecast panels."""
    sub_html = f"<div class='sim-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='sim-card'>"
        f"<div class='sim-val' style='color:{color}'>{val}</div>"
        f"<div class='sim-lbl'>{label}</div>"
        f"{sub_html}</div>"
    )
