# Urban-Mobility-Intelligence-System

Turning 34.5 million NYC taxi trips into ETA predictions you can trust, demand forecasts you can plan around, and routes you can benchmark, all in one interactive dashboard.

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM ETA Model](https://img.shields.io/badge/LightGBM-ETA%20Model-success)
![PySpark Demand Forecasting](https://img.shields.io/badge/PySpark-Demand%20Forecasting-orange)
![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)

# The Problem

Taxi fleets generate massive amounts of trip data, yet dispatch decisions are still mostly reactive. Riders get ETAs with no sense of how reliable they are, and drivers are sent to demand after it appears. This project asks three operational questions:

1. How long will this trip take, and how confident should we be?
  
3. Where and when will demand spike next?
   
4. How efficient are the routes drivers actually take?
   
# Highlights

• ETA prediction at scale. LightGBM trained on 21.7M trips, evaluated on 11.9M unseen trips: MAE 2.3 min, R² 0.861.

• Outperformed four alternatives: XGBoost, CatBoost, GAM and a stacked hybrid ensemble.

• Reliability scoring layer. Every prediction is labelled High, Medium or Low reliability. High-reliability trips (92.6%) average 2.1 min error; flagged trips average 4.9 min, so dispatchers know when to add a buffer.

• Demand forecasting with PySpark. Gradient Boosted Trees forecast hourly demand across 2,847 city zones (R² 0.84), using lag and rolling-window features.
Road-network route simulation. Dijkstra shortest paths on Manhattan's real street graph (OSMnx) benchmark around 1,000 trips. Actual routes average 1.64x the optimal distance.

# Results: ETA Model Comparison

Tested on March 2016 (unseen data):

Model	| Training data | MAE (min)	R²

LightGBM v3 (selected) | 21.7M trips | 2.306	0.861

LightGBM | 5M sample | 2.684	0.828

XGBoost	| 5M sample	| 2.700	0.828

CatBoost	| 5M sample	| 2.744	0.823

GAM	| 200K sample	| 3.138	0.699


LightGBM leads even when every model is trained on the same 5M sample, which is why it was the one scaled to the full dataset.

# Error analysis insights

• Airport pickups are about 2x harder to predict (4.5 min vs 2.2 min MAE). This is the model's biggest weak spot and the reason the reliability layer exists.

• Weekends are easier to predict than weekdays (2.1 vs 2.4 min), and 5 AM is the most predictable hour.

• Error grows with distance, from 1.3 min on short trips to 6.7 min on 20+ mile trips.

# Approach

NYC TLC trip records (Jan–Mar 2016, 34.5M trips)
   │
   
   ├─ Cleaning ............. coordinate outliers, impossible durations/distances (2.6% removed)
   
   ├─ Feature engineering .. 28 features: temporal, spatial, airport flags, zone clusters,
   
   │                         congestion memory (zone speed 15/30/60 min earlier), target encoding
   
   ├─ ETA model ............ LightGBM, temporal split (train Jan–Feb → test Mar)
   
   ├─ Reliability layer .... rule-based score from distance, airport, rush hour, zone
   
   ├─ Demand model ......... PySpark GBT on zone-hour aggregates with lag features
   
   └─ Route simulation ..... OSMnx graph + Dijkstra shortest path vs actual trip distance


# Dashboard

A four-section Streamlit app:

Section |	What you can do

Data Overview & EDA: Explore speed patterns by hour and zone, weekday vs weekend, anomalies.

ETA Prediction: Predicted vs actual, error breakdowns, What-If ETA simulator, trip lookup.

Demand Prediction:	Hourly demand curves, feature importance, forecast playground.

Path Simulation: Live shortest-path finder on an interactive map, route comparisons.




# Repository Structure

├── app.py                 # Streamlit entry point

├── data_loader.py         # cached data loading

├── seg_eda.py             # Section 1: EDA

├── seg_eta.py             # Section 2: ETA prediction

├── seg_demand.py          # Section 3: demand prediction

├── seg_path.py            # Section 4: path simulation

├── path_engine.py         # OSMnx + Dijkstra routing engine

├── genai.py               # LLM Q&A panel

├── styles.py              # theme and shared UI helpers

├── notebooks/             # cleaning, EDA, ETA modelling, PySpark demand, path simulation

├── data/                  # sample CSVs (full data not tracked)

└── requirements.txt


The dashboard runs on the bundled sample data (data/path_1000.csv, data/path_20.csv).

# Data

Source: NYC Yellow Taxi Trip Data, January–March 2016 

[https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data?select=yellow_tripdata_2016-03.csv]

Full trip data and prediction files (about 900 MB) are not stored in this repo. Download the raw data from the link above.

# Tech Stack

ML & Data: Python, Pandas, NumPy, LightGBM, XGBoost, CatBoost, scikit-learn, PyArrow 

Big Data: Apache PySpark (MLlib) 

Geospatial: OSMnx, NetworkX, Folium App: Streamlit, Plotly
