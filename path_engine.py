"""path_engine.py – Real-time path computation using OSMnx + NetworkX.

Provides functions to:
  • Load / download the Manhattan road network graph
  • Compute shortest path between two GPS coordinates
  • Return path polyline for map rendering
"""

import os
import streamlit as st
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GRAPH_FILE = os.path.join(DATA_DIR, "nyc_manhattan.graphml")

# ── Pre-defined NYC landmarks for the picker ─────────────────────────────────
LANDMARKS = {
    "Times Square":         (40.7580, -73.9855),
    "Penn Station":         (40.7506, -73.9935),
    "Grand Central":        (40.7527, -73.9772),
    "JFK Airport":          (40.6413, -73.7781),
    "LaGuardia Airport":    (40.7769, -73.8740),
    "Central Park South":   (40.7648, -73.9724),
    "Wall Street":          (40.7074, -74.0113),
    "Empire State":         (40.7484, -73.9857),
    "Brooklyn Bridge":      (40.7061, -73.9969),
    "Columbus Circle":      (40.7681, -73.9819),
    "Union Square":         (40.7359, -73.9911),
    "SoHo":                 (40.7233, -73.9985),
    "East Village":         (40.7265, -73.9815),
    "Upper East Side":      (40.7736, -73.9566),
    "Harlem":               (40.8116, -73.9465),
    "Chelsea":              (40.7465, -74.0014),
    "Greenwich Village":    (40.7336, -74.0027),
    "Midtown East":         (40.7549, -73.9680),
}


@st.cache_resource(show_spinner="Loading NYC road network (first time may take ~60s)…")
def load_graph():
    """Load or download the Manhattan road network graph.

    The graph is cached to disk as a .graphml file after first download
    and kept in memory via @st.cache_resource.
    """
    import osmnx as ox

    if os.path.exists(GRAPH_FILE):
        G = ox.load_graphml(GRAPH_FILE)
    else:
        # Download the drivable road network for Manhattan + nearby areas
        G = ox.graph_from_place("Manhattan, New York, USA", network_type="drive")
        os.makedirs(DATA_DIR, exist_ok=True)
        ox.save_graphml(G, GRAPH_FILE)
    return G


def compute_shortest_path(G, orig_lat: float, orig_lon: float,
                          dest_lat: float, dest_lon: float) -> dict:
    """Compute the shortest path between two GPS points on the road network.

    Returns dict with keys:
        path_coords: list of (lat, lon) tuples for the route polyline
        distance_km: total shortest-path distance in km
        haversine_km: straight-line distance in km
        route_efficiency: haversine / shortest (1.0 = perfectly direct)
        num_nodes: number of nodes in the path
        orig_node: nearest node to origin
        dest_node: nearest node to destination
        error: None or error message string
    """
    import osmnx as ox
    import networkx as nx

    result = {
        "path_coords": [],
        "distance_km": 0.0,
        "haversine_km": 0.0,
        "route_efficiency": 0.0,
        "num_nodes": 0,
        "orig_node": None,
        "dest_node": None,
        "error": None,
    }

    try:
        # Snap GPS to nearest graph nodes
        orig_node = ox.nearest_nodes(G, orig_lon, orig_lat)
        dest_node = ox.nearest_nodes(G, dest_lon, dest_lat)
        result["orig_node"] = orig_node
        result["dest_node"] = dest_node

        if orig_node == dest_node:
            result["error"] = "Pickup and dropoff snap to the same road node. Try points farther apart."
            return result

        # Compute shortest path (Dijkstra, weight=length in metres)
        path_nodes = nx.shortest_path(G, orig_node, dest_node, weight="length")
        path_length_m = nx.shortest_path_length(G, orig_node, dest_node, weight="length")

        result["num_nodes"] = len(path_nodes)
        result["distance_km"] = path_length_m / 1000.0

        # Build polyline from node coordinates
        coords = []
        for node in path_nodes:
            node_data = G.nodes[node]
            coords.append((node_data["y"], node_data["x"]))  # (lat, lon)
        result["path_coords"] = coords

        # Haversine distance
        result["haversine_km"] = _haversine(orig_lat, orig_lon, dest_lat, dest_lon)

        # Route efficiency
        if result["haversine_km"] > 0:
            result["route_efficiency"] = result["distance_km"] / result["haversine_km"]

    except nx.NetworkXNoPath:
        result["error"] = "No path found between these points. They may be on disconnected road segments."
    except Exception as e:
        result["error"] = f"Path computation error: {str(e)}"

    return result


def _haversine(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km between two GPS points."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))
