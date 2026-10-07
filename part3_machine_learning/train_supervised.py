"""
Part 3: Task 1 - Supervised Machine Learning Pipeline
Trains regression and classification models with standard logging and metric reporting.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Standard logging configuration matching Part 2
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


def prepare_data(data_path):
    logger.info(f"Loading raw dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])

    # 1. Temporal Features
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    df["is_holiday"] = (df["holiday"].fillna("None") != "None").astype(int)

    # 2. Weather Encodings
    low_vis = ["Fog", "Mist", "Smoke", "Haze"]
    severe_weather = ["Squall", "Thunderstorm", "Snow"]
    df["is_low_visibility"] = df["weather_main"].isin(low_vis).astype(int)

    # 3. Congestion Quartile Bucketing
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values

    def bucket(v):
        if v <= q1:
            return "Low"
        elif v <= q2:
            return "Medium"
        elif v <= q3:
            return "High"
        return "Severe"

    df["congestion_category"] = df["traffic_volume"].apply(bucket)

    # 4. Proxy Accident-Risk Label
    high_congestion = df["congestion_category"].isin(["High", "Severe"])
    risky_weather = df["weather_main"].isin(severe_weather) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (high_congestion & risky_weather).astype(int)

    # 5. Temperature Outlier Imputation
    if (df["temp"] == 0).any():
        median_temp = df.loc[df["temp"] > 0, "temp"].median()
        df["temp"] = df["temp"].replace(0, median_temp)
        logger.warning(f"Imputed 0 Kelvin temperature outliers with median: {median_temp:.2f} K")

    logger.info(
        f"Data preparation complete. Shape: {df.shape}, "
        f"High-risk instances: {df['high_risk'].sum()} ({df['high_risk'].mean():.2%})"
    )
    return df


def assemble_features(df):
    numeric_features = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_holiday", "temp", "rain_1h", 
        "snow_1h", "clouds_all", "is_low_visibility"
    ]
    weather_encoded = pd.get_dummies(df["weather_main"], prefix="weather", drop_first=True)
    X = pd.concat([df[numeric_features], weather_encoded], axis=1)
    y_reg = df["traffic_volume"]
    y_clf = df["high_risk"]
    return X, y_reg, y_clf


def run():
    setup_logging()
    logger.info("==================================================")
    logger.info("Starting Supervised ML Pipeline Execution")
    logger.info("==================================================")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(root_dir, "Metro_Interstate_Traffic_Volume.csv")
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = prepare_data(data_path)
    X, y_reg, y_clf = assemble_features(df)

    # ----------------------------------------------------
    # Task 1A: Regression
    # ----------------------------------------------------
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.20, random_state=42
    )

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_r, y_train_r)
    mae_lr = mean_absolute_error(y_test_r, lin_reg.predict(X_test_r))
    r2_lr = r2_score(y_test_r, lin_reg.predict(X_test_r))
    logger.info(f"Linear Regression Baseline -> MAE: {mae_lr:.2f} vehicles, R2: {r2_lr:.4f}")

    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf_reg.fit(X_train_r, y_train_r)
    mae_rf = mean_absolute_error(y_test_r, rf_reg.predict(X_test_r))
    r2_rf = r2_score(y_test_r, rf_reg.predict(X_test_r))
    logger.info(f"Random Forest Regressor -> MAE: {mae_rf:.2f} vehicles, R2: {r2_rf:.4f}")

    joblib.dump(rf_reg, os.path.join(models_dir, "best_regression_rf.pkl"))
    logger.info("Saved model artifact: models/best_regression_rf.pkl")

    # ----------------------------------------------------
    # Task 1B: Classification
    # ----------------------------------------------------
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    log_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    log_pipe.fit(X_train_c, y_train_c)
    y_pred_log = log_pipe.predict(X_test_c)
    logger.info(
        f"Logistic Regression Baseline -> Accuracy: {accuracy_score(y_test_c, y_pred_log):.4f}, "
        f"Recall: {recall_score(y_test_c, y_pred_log):.4f}, F1: {f1_score(y_test_c, y_pred_log):.4f}"
    )

    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf_clf.fit(X_train_c, y_train_c)
    y_pred_rf = rf_clf.predict(X_test_c)
    logger.info(
        f"Random Forest Classifier -> Accuracy: {accuracy_score(y_test_c, y_pred_rf):.4f}, "
        f"Precision: {precision_score(y_test_c, y_pred_rf, zero_division=0):.4f}, "
        f"Recall: {recall_score(y_test_c, y_pred_rf):.4f}, "
        f"F1: {f1_score(y_test_c, y_pred_rf):.4f}, "
        f"ROC AUC: {roc_auc_score(y_test_c, rf_clf.predict_proba(X_test_c)[:, 1]):.4f}"
    )

    joblib.dump(rf_clf, os.path.join(models_dir, "best_classification_rf.pkl"))
    logger.info("Saved model artifact: models/best_classification_rf.pkl")
    logger.info("Supervised ML pipeline execution complete.")


if __name__ == "__main__":
    run()