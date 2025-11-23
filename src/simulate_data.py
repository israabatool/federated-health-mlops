# src/simulate_data.py
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def simulate_node(node_name, n_samples=5000, seed=0):
    rng = np.random.RandomState(seed)
    # timestamps (optional)
    times = pd.date_range("2025-01-01", periods=n_samples, freq="T")  # 1-minute windows
    # wearable signals
    heart_rate = rng.normal(loc=75, scale=10, size=n_samples).clip(40, 200)
    steps = rng.poisson(lam=50, size=n_samples)
    spo2 = rng.normal(loc=97, scale=1.5, size=n_samples).clip(80, 100)
    # air quality
    pm25 = np.abs(rng.normal(loc=40 + seed*5, scale=20, size=n_samples))  # vary per node
    pm10 = pm25 + np.abs(rng.normal(loc=5, scale=5, size=n_samples))
    co2 = rng.normal(loc=400, scale=50, size=n_samples).clip(200, 2000)
    # weather
    temp = rng.normal(loc=30 - seed, scale=3, size=n_samples)
    humidity = rng.uniform(20, 90, size=n_samples)
    # label (toy rule): high risk if pm25 > 75 or heart_rate > 110 and spo2 < 94
    risk = ((pm25 > 75) | ((heart_rate > 110) & (spo2 < 94))).astype(int)

    df = pd.DataFrame({
        "timestamp": times,
        "heart_rate": np.round(heart_rate, 2),
        "steps": steps,
        "spo2": np.round(spo2, 2),
        "pm25": np.round(pm25, 2),
        "pm10": np.round(pm10, 2),
        "co2": np.round(co2, 2),
        "temp": np.round(temp, 2),
        "humidity": np.round(humidity, 2),
        "risk": risk
    })
    out_path = DATA_DIR / f"{node_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")
    return out_path

def generate_all(num_nodes=3, samples_per_node=5000):
    nodes = []
    for i in range(num_nodes):
        name = f"node_{chr(ord('A') + i)}"
        path = simulate_node(name, n_samples=samples_per_node, seed=i+1)
        nodes.append(path)
    # Write a simple manifest
    manifest = {"nodes": [p.name for p in nodes]}
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Generated manifest.json")
    return nodes

if __name__ == "__main__":
    generate_all(num_nodes=3, samples_per_node=3000)

