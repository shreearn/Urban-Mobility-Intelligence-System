"""
data_loader.py
==============
Centralised data loading for the NYC Taxi Dashboard.

Parquet files (large, optional – place in data/ folder):
  • taxi_2016_test_final_with_predictions.parquet  → ETA segment
  • final_results_demand_pred_2016.parquet          → Demand segment

CSV files (always required – already bundled in data/):
  • path_1000.csv   – 999 rows, all 42 features + shortest-path metrics
  • path_20.csv     – 20 rows for interactive route map

The loader gracefully falls back to the CSVs if parquet files are missing.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── file paths ────────────────────────────────────────────────────────────────
PATH_1000  = os.path.join(DATA_DIR, "path_1000.csv")
PATH_20    = os.path.join(DATA_DIR, "path_20.csv")
ETA_PARQ   = os.path.join(DATA_DIR, "taxi_2016_test_final_with_predictions.parquet")
DEM_PARQ   = os.path.join(DATA_DIR, "final_results_demand_pred_2016.parquet")


# ── helper: derive distance_bin ───────────────────────────────────────────────
def _ensure_distance_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Add actual_km and distance_bin columns if they are missing."""
    if "actual_km" not in df.columns and "trip_distance" in df.columns:
        df["actual_km"] = df["trip_distance"] * 1.60934
    if "distance_bin" not in df.columns and "actual_km" in df.columns:
        df["distance_bin"] = pd.cut(
            df["actual_km"],
            bins=[0, 2, 5, 10, float("inf")],
            labels=["Short (0-2 km)", "Medium (2-5 km)",
                     "Long (5-10 km)", "Very Long (10+ km)"],
        ).astype(str)
    return df


# ── loaders ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading path data…")
def load_path_1000() -> pd.DataFrame:
    df = pd.read_csv(PATH_1000, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df["abs_err"] = df["prediction_error_min"].abs()
    return df


@st.cache_data(show_spinner="Loading 20-sample map data…")
def load_path_20() -> pd.DataFrame:
    return pd.read_csv(PATH_20, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])


@st.cache_data(show_spinner="Loading ETA parquet (may take a moment)…")
def load_eta_parquet(sample_n: int = 50_000) -> tuple[pd.DataFrame, bool]:
    """
    Returns (dataframe, is_full_parquet).
    If the parquet is present uses it (sampled to sample_n for performance).
    Otherwise falls back to path_1000.csv with is_full_parquet=False.
    """
    if os.path.exists(ETA_PARQ):
        df = pd.read_parquet(ETA_PARQ)
        # ensure column naming matches CSV columns
        rename = {
            "lgbm_v3_pred_min": "lgbm_v3_pred_min",
            "actual_duration_min": "actual_duration_min",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "prediction_error_min" not in df.columns and "lgbm_v3_pred_min" in df.columns:
            df["prediction_error_min"] = df["lgbm_v3_pred_min"] - df["actual_duration_min"]
        if "abs_err" not in df.columns:
            df["abs_err"] = df["prediction_error_min"].abs()
        if "reliability_label" not in df.columns and "reliability_score" in df.columns:
            df["reliability_label"] = pd.cut(
                df["reliability_score"], bins=[-1, 49, 74, 100],
                labels=["Low", "Medium", "High"]
            ).astype(str)
        df = _ensure_distance_bin(df)
        if len(df) > sample_n:
            df = df.sample(sample_n, random_state=42).reset_index(drop=True)
        return df, True
    else:
        df = load_path_1000()
        return df, False


@st.cache_data(show_spinner="Building ETA lookup table…")
def load_eta_lookup() -> pd.DataFrame:
    """Pre-aggregated lookup table for the What-If ETA Simulator.

    Groups the full ETA parquet by (pickup_hour, pickup_cluster, distance_bin,
    rush_hour, is_weekend) and aggregates prediction statistics.
    Falls back to path_1000.csv if parquet is missing.
    """
    if os.path.exists(ETA_PARQ):
        df = pd.read_parquet(ETA_PARQ)
    else:
        df = pd.read_csv(PATH_1000,
                         parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

    # Ensure required columns
    if "prediction_error_min" not in df.columns and "lgbm_v3_pred_min" in df.columns:
        df["prediction_error_min"] = df["lgbm_v3_pred_min"] - df["actual_duration_min"]
    df["abs_err"] = df["prediction_error_min"].abs()
    df = _ensure_distance_bin(df)

    group_cols = ["pickup_hour", "pickup_cluster", "distance_bin", "rush_hour", "is_weekend"]
    lookup = df.groupby(group_cols, observed=True).agg(
        avg_pred=("lgbm_v3_pred_min", "mean"),
        avg_actual=("actual_duration_min", "mean"),
        avg_error=("prediction_error_min", "mean"),
        mae=("abs_err", "mean"),
        std_pred=("lgbm_v3_pred_min", "std"),
        count=("lgbm_v3_pred_min", "count"),
        avg_distance=("trip_distance", "mean"),
    ).reset_index()
    lookup["std_pred"] = lookup["std_pred"].fillna(0)
    return lookup


@st.cache_data(show_spinner="Loading demand parquet…")
def load_demand_parquet(sample_n: int = 100_000) -> tuple[pd.DataFrame, bool]:
    """
    Returns (dataframe, is_full_parquet).
    Falls back to synthetic representative data if parquet missing.
    """
    if os.path.exists(DEM_PARQ):
        df = pd.read_parquet(DEM_PARQ)
        if len(df) > sample_n:
            df = df.sample(sample_n, random_state=42).reset_index(drop=True)
        return df, True
    return pd.DataFrame(), False   # caller handles empty case


@st.cache_data(show_spinner="Building demand lookup table…")
def load_demand_lookup() -> pd.DataFrame:
    """Pre-aggregated demand lookup for the Forecast Playground."""
    if not os.path.exists(DEM_PARQ):
        return pd.DataFrame()

    df = pd.read_parquet(DEM_PARQ)
    group_cols = ["zone_type", "day_of_week", "hour", "is_peak_hour", "is_weekend"]
    lookup = df.groupby(group_cols, observed=True).agg(
        avg_demand=("demand", "mean"),
        avg_prediction=("final_prediction", "mean"),
        std_demand=("demand", "std"),
        count=("demand", "count"),
        avg_lag1=("lag_1", "mean"),
        avg_rolling3=("rolling_3", "mean"),
    ).reset_index()
    lookup["std_demand"] = lookup["std_demand"].fillna(0)
    return lookup


def parquet_status() -> dict:
    """Returns dict of parquet availability for status display."""
    return {
        "eta":    os.path.exists(ETA_PARQ),
        "demand": os.path.exists(DEM_PARQ),
        "eta_path":    ETA_PARQ,
        "demand_path": DEM_PARQ,
    }
