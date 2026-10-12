import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader

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
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

class TrafficLSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_layers=2, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc_head(last_out).squeeze(-1)


class TrafficSequenceDataset(Dataset):
    def __init__(self, features, targets, seq_len=24):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_and_preprocess_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "part2_python", "traffic_features.csv")
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)

    # Time-based and cyclical features
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = (df["holiday"] != "None").astype(int) if "holiday" in df.columns else 0

    low_vis = ["Fog", "Mist", "Smoke", "Haze"]
    df["is_low_visibility"] = df["weather_main"].isin(low_vis).astype(int) if "weather_main" in df.columns else 0

    weather_cats = ["Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]
    for w in weather_cats:
        df[f"weather_{w}"] = (df["weather_main"] == w).astype(int) if "weather_main" in df.columns else 0

    feature_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
        "temp", "rain_1h", "snow_1h", "clouds_all", "is_low_visibility"
    ] + [f"weather_{w}" for w in weather_cats]
    target_col = "traffic_volume"

    # Chronological Split (70% train, 15% validation, 15% test)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df, feature_cols, target_col

if __name__ == "__main__":
    setup_logging()
    set_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    experiment_name = "SmartCity_Traffic_Volume_Forecasting"
    mlflow.set_experiment(experiment_name)
    logger.info(f"Tracking MLflow experiments at: {db_path}")

    train_df, val_df, test_df, feature_cols, target_col = load_and_preprocess_data()

    # Preprocessing
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(train_df[feature_cols])
    X_test = scaler_x.transform(test_df[feature_cols])

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(train_df[[target_col]]).flatten()
    y_test = scaler_y.transform(test_df[[target_col]]).flatten()
    y_test_unscaled = test_df[target_col].values

    # ==========================================
    # Run 1: Linear Regression (Baseline)
    # ==========================================
    logger.info("[MLflow Run 1] Tracking Baseline Linear Regression...")
    with mlflow.start_run(run_name="Baseline_Linear_Regression"):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_preds_unscaled = scaler_y.inverse_transform(lr_preds.reshape(-1, 1)).flatten()

        lr_mae = mean_absolute_error(y_test_unscaled, lr_preds_unscaled)
        lr_rmse = np.sqrt(mean_squared_error(y_test_unscaled, lr_preds_unscaled))
        lr_r2 = r2_score(y_test_unscaled, lr_preds_unscaled)

        # Log Parameters
        mlflow.log_param("model_family", "LinearModel")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("num_features", len(feature_cols))

        # Log Metrics
        mlflow.log_metric("test_mae", lr_mae)
        mlflow.log_metric("test_rmse", lr_rmse)
        mlflow.log_metric("test_r2", lr_r2)

        # Log Model Artifact
        mlflow.sklearn.log_model(lr, name="model")
        logger.info(f"Linear Regression logged -> MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}, R²: {lr_r2:.4f}")
        
    # ==========================================
    # Run 2: Random Forest Regressor
    # ==========================================
    logger.info("[MLflow Run 2] Tracking Random Forest Regressor...")
    with mlflow.start_run(run_name="Random_Forest_Regressor"):
        n_estimators = 100
        max_depth = 18
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_preds_unscaled = scaler_y.inverse_transform(rf_preds.reshape(-1, 1)).flatten()

        rf_mae = mean_absolute_error(y_test_unscaled, rf_preds_unscaled)
        rf_rmse = np.sqrt(mean_squared_error(y_test_unscaled, rf_preds_unscaled))
        rf_r2 = r2_score(y_test_unscaled, rf_preds_unscaled)

        # Log Parameters
        mlflow.log_param("model_family", "TreeEnsemble")
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("num_features", len(feature_cols))

        # Log Metrics
        mlflow.log_metric("test_mae", rf_mae)
        mlflow.log_metric("test_rmse", rf_rmse)
        mlflow.log_metric("test_r2", rf_r2)

        # Log Model Artifact
        mlflow.sklearn.log_model(rf, name="model")
        logger.info(f"Random Forest logged -> MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}")
        
    # ==========================================
    # Run 3: PyTorch LSTM (Sequential Deep Learning)
    # ==========================================
    logger.info("[MLflow Run 3] Tracking PyTorch 2-Layer LSTM...")
    with mlflow.start_run(run_name="PyTorch_Stacked_LSTM"):
        seq_len = 24
        hidden_dim = 64
        num_layers = 2
        dropout = 0.2

        best_lstm_path = os.path.join(script_dir, "models", "best_traffic_lstm.pt")
        lstm_model = TrafficLSTM(input_dim=len(feature_cols), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        checkpoint = torch.load(best_lstm_path, map_location="cpu")
        lstm_model.load_state_dict(checkpoint["model_state_dict"])
        lstm_model.eval()

        test_ds = TrafficSequenceDataset(X_test, y_test, seq_len=seq_len)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        preds_list, acts_list = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                preds_list.extend(lstm_model(x_b).numpy())
                acts_list.extend(y_b.numpy())

        lstm_preds_unscaled = scaler_y.inverse_transform(np.array(preds_list).reshape(-1, 1)).flatten()
        lstm_acts_unscaled = scaler_y.inverse_transform(np.array(acts_list).reshape(-1, 1)).flatten()

        lstm_mae = mean_absolute_error(lstm_acts_unscaled, lstm_preds_unscaled)
        lstm_rmse = np.sqrt(mean_squared_error(lstm_acts_unscaled, lstm_preds_unscaled))
        lstm_r2 = r2_score(lstm_acts_unscaled, lstm_preds_unscaled)

        # Log Parameters
        mlflow.log_param("model_family", "RecurrentNeuralNetwork")
        mlflow.log_param("model_type", "PyTorch_LSTM")
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("dropout", dropout)

        # Log Metrics
        mlflow.log_metric("test_mae", lstm_mae)
        mlflow.log_metric("test_rmse", lstm_rmse)
        mlflow.log_metric("test_r2", lstm_r2)

        # Log Model Artifact with an input example (batch_size=1, seq_len=24, num_features=21)
        example_input = torch.zeros((1, seq_len, len(feature_cols)), dtype=torch.float32)
        mlflow.pytorch.log_model(
            pytorch_model=lstm_model,
            artifact_path="model",
            input_example=example_input.numpy()
        )
        logger.info(f"PyTorch LSTM logged -> MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}, R²: {lstm_r2:.4f}")

    logger.info("All 3 candidate models logged to MLflow successfully.")