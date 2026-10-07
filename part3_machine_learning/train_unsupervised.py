"""
Part 3: Task 2 - Unsupervised Machine Learning Pipeline
Executes K-Means Clustering and Association Rule Mining with logging and artifact exports.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Module-level logger setup
logger = logging.getLogger(__name__)


def setup_logging():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "ml_pipeline.log")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False


def load_and_preprocess(data_path):
    logger.info(f"Loading raw dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])

    # 1. Feature extraction
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 2. Temperature outlier handling (replace 0 K with median)
    if (df["temp"] == 0).any():
        median_temp = df.loc[df["temp"] > 0, "temp"].median()
        df["temp"] = df["temp"].replace(0, median_temp)
        logger.warning(f"Imputed 0 Kelvin temperature outliers with median: {median_temp:.2f} K")

    # 3. Rain sensor anomaly handling (cap extreme outlier > 100mm)
    rain_anomalies = (df["rain_1h"] > 100).sum()
    if rain_anomalies > 0:
        logger.warning(f"Capping {rain_anomalies} extreme rain_1h outlier(s) to 100.0 mm")
        df["rain_1h_capped"] = df["rain_1h"].clip(upper=100.0)
    else:
        df["rain_1h_capped"] = df["rain_1h"]

    # 4. Congestion category derivation
    df["congestion_category"] = pd.qcut(
        df["traffic_volume"],
        q=4,
        labels=["Low", "Medium", "High", "Severe"]
    )

    logger.info(f"Preprocessing complete. Dataset shape: {df.shape}")
    return df


def run_clustering(df, models_dir):
    logger.info("Executing K-Means clustering (k=3)...")
    cluster_cols = ["traffic_volume", "hour", "day_of_week", "temp", "rain_1h_capped"]

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df[cluster_cols])

    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled_matrix)

    # Persist clustering model and scaler artifacts
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_traffic_regimes.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler_kmeans.pkl"))
    logger.info("Saved clustering artifacts to 'models/'")

    # Map readable regimes based on mean traffic
    profile = df.groupby("cluster")["traffic_volume"].mean().sort_values().reset_index()
    regime_mapping = {
        profile.loc[0, "cluster"]: "Off-Peak / Night Lull",
        profile.loc[1, "cluster"]: "Moderate / Off-Peak Daytime",
        profile.loc[2, "cluster"]: "Peak Commute / Rush Hour"
    }
    df["traffic_regime"] = df["cluster"].map(regime_mapping)

    for cluster_id, regime_name in regime_mapping.items():
        count = (df["cluster"] == cluster_id).sum()
        avg_vol = df.loc[df["cluster"] == cluster_id, "traffic_volume"].mean()
        logger.info(f"Cluster {cluster_id} ({regime_name}) -> Count: {count:,} ({count/len(df):.1%}), Mean Volume: {avg_vol:.1f}")

    return df


def run_association_rules(df, output_dir):
    logger.info("Executing Association Rule Mining via FP-Growth...")

    # Time slot bucketing
    def time_bucket(hour):
        if 6 <= hour <= 9:
            return "Time_MorningRush"
        elif 10 <= hour <= 15:
            return "Time_Midday"
        elif 16 <= hour <= 19:
            return "Time_EveningRush"
        else:
            return "Time_NightOffPeak"

    df["time_slot"] = df["hour"].apply(time_bucket)
    df["congestion_label"] = "Congestion_" + df["congestion_category"].astype(str)
    df["day_label"] = np.where(df["is_weekend"] == 1, "Day_Weekend", "Day_Weekday")
    df["weather_label"] = "Weather_" + df["weather_main"].astype(str)
    df["regime_label"] = "Regime_" + df["traffic_regime"].str.replace(" ", "")

    transaction_cols = ["time_slot", "congestion_label", "weather_label", "day_label", "regime_label"]
    basket = pd.get_dummies(df[transaction_cols], prefix="", prefix_sep="").astype(bool)

    frequent_itemsets = fpgrowth(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.60)
    filtered_rules = rules[rules["lift"] > 1.2].sort_values(by="lift", ascending=False).reset_index(drop=True)

    rules_path = os.path.join(output_dir, "traffic_association_rules.csv")
    filtered_rules.to_csv(rules_path, index=False)
    logger.info(f"Mined {len(filtered_rules)} rules (min_support=0.05, min_confidence=0.60, lift>1.2). Saved to '{rules_path}'")


def run():
    setup_logging()
    logger.info("==================================================")
    logger.info("Starting Unsupervised ML Pipeline Execution")
    logger.info("==================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(root_dir, "Metro_Interstate_Traffic_Volume.csv")
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = load_and_preprocess(data_path)
    df = run_clustering(df, models_dir)
    run_association_rules(df, current_dir)

    logger.info("Unsupervised ML pipeline execution complete.")


if __name__ == "__main__":
    run()