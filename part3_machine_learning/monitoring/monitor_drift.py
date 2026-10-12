"""
Part 3 - Task 6.4: Model Monitoring Simulation
Simulates production monitoring checking for:
  1. Feature distribution drift (temp_c, clouds_all)
  2. Prediction / target error drift (traffic_volume)
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def load_data_and_model():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, "..", ".."))

    # Baseline dataset
    baseline_csv = os.path.join(project_root, "part2_python", "traffic_features.csv")
    if not os.path.exists(baseline_csv):
        baseline_csv = os.path.join(project_root, "Metro_Interstate_Traffic_Volume.csv")
    df_base = pd.read_csv(baseline_csv)

    if "temp" in df_base.columns and df_base["temp"].mean() > 150:
        df_base["temp_c"] = df_base["temp"] - 273.15
    else:
        df_base["temp_c"] = df_base["temp"]

    # Champion model for generating production predictions
    model_path = os.path.join(curr_dir, "..", "models", "best_regression_rf.pkl")
    model = joblib.load(model_path) if os.path.exists(model_path) else None

    return df_base, model

def simulate_production_windows(df_base, n_samples=300):
    np.random.seed(42)

    # Window 1: In-distribution normal operations
    win1_temp = np.random.normal(df_base["temp_c"].mean(), df_base["temp_c"].std(), n_samples)
    win1_clouds = np.random.choice(df_base["clouds_all"].dropna(), n_samples)
    win1_vol = np.random.choice(df_base["traffic_volume"].dropna(), n_samples)
    df_win1 = pd.DataFrame({"temp_c": win1_temp, "clouds_all": win1_clouds, "traffic_volume": win1_vol})

    # Window 2: Out-of-distribution drift (polar freeze + atypical volume surge)
    win2_temp = np.random.normal(-22.0, 3.5, n_samples)  # Severe freeze
    win2_clouds = np.random.normal(92.0, 4.0, n_samples)
    win2_vol = np.random.normal(6100.0, 350.0, n_samples) # Heavy sustained surge
    df_win2 = pd.DataFrame({"temp_c": win2_temp, "clouds_all": win2_clouds, "traffic_volume": win2_vol})

    return df_win1, df_win2

def check_drift(window_name, df_base, df_prod, alpha=0.01):
    print("\n" + "=" * 82)
    print(f"MONITORING SIMULATION: {window_name}")
    print("=" * 82)

    metrics = [
        ("temp_c", "Feature: Temperature (°C)"),
        ("clouds_all", "Feature: Cloud Cover (%)"),
        ("traffic_volume", "Prediction / Target Volume")
    ]

    records = []
    for col, label in metrics:
        base_vals = df_base[col].dropna().values
        prod_vals = df_prod[col].dropna().values

        ks_stat, p_val = ks_2samp(base_vals, prod_vals)
        drift_detected = bool(p_val < alpha)

        records.append({
            "Monitored Entity": label,
            "Baseline Mean": round(float(np.mean(base_vals)), 2),
            "Prod Window Mean": round(float(np.mean(prod_vals)), 2),
            "KS Statistic": round(float(ks_stat), 4),
            "p-value": f"{p_val:.2e}",
            "Drift Detected": "YES (Drifted)" if drift_detected else "NO (Aligned)"
        })

    df_res = pd.DataFrame(records)
    print(df_res.to_string(index=False))
    print("-" * 82)
    return df_res

if __name__ == "__main__":
    df_base, _ = load_data_and_model()
    df_win1, df_win2 = simulate_production_windows(df_base)
    check_drift("Window 1 (Standard Operations)", df_base, df_win1)
    check_drift("Window 2 (Extreme Winter & Volume Shift)", df_base, df_win2)