"""
Part 3 - Task 3: Deep Learning with Explainability
End-to-end training and explainability pipeline for sequential traffic modeling using PyTorch LSTM & SHAP.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrafficLSTM(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, num_layers=2, dropout=0.2):
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


class ShapLSTMWrapper(nn.Module):
    def __init__(self, base_model):
        super(ShapLSTMWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


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


def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing Deep Learning Pipeline on Device: {device}")

    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "part2_python", "traffic_features.csv")
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)

    # 2. Features
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

    # Splits
    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    # Scaling
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(train_df[feature_cols])
    X_val = scaler_x.transform(val_df[feature_cols])
    X_test = scaler_x.transform(test_df[feature_cols])

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(train_df[[target_col]]).flatten()
    y_val = scaler_y.transform(val_df[[target_col]]).flatten()
    y_test = scaler_y.transform(test_df[[target_col]]).flatten()

    SEQ_LEN = 24
    train_ds = TrafficSequenceDataset(X_train, y_train, seq_len=SEQ_LEN)
    val_ds = TrafficSequenceDataset(X_val, y_val, seq_len=SEQ_LEN)
    test_ds = TrafficSequenceDataset(X_test, y_test, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 3. Model & Training
    model = TrafficLSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_path = os.path.join(models_dir, "best_traffic_lstm.pt")

    best_val_loss = float("inf")
    patience, patience_cnt = 5, 0

    print("Beginning model training...")
    for epoch in range(1, 21):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                val_loss += criterion(model(x_b), y_b).item() * len(y_b)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(), "best_val_loss": best_val_loss}, best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 4. CPU Evaluation
    eval_model = TrafficLSTM(input_dim=len(feature_cols)).to("cpu")
    ckpt = torch.load(best_path, map_location="cpu")
    eval_model.load_state_dict(ckpt["model_state_dict"])
    eval_model.eval()

    preds_list, acts_list = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            preds_list.extend(eval_model(x_b.to("cpu")).numpy())
            acts_list.extend(y_b.numpy())

    preds = scaler_y.inverse_transform(np.array(preds_list).reshape(-1, 1)).flatten()
    acts = scaler_y.inverse_transform(np.array(acts_list).reshape(-1, 1)).flatten()

    mae = mean_absolute_error(acts, preds)
    rmse = np.sqrt(mean_squared_error(acts, preds))
    r2 = r2_score(acts, preds)

    print("\n--- Final Test Results ---")
    print(f"MAE  : {mae:.2f} vehicles/hr")
    print(f"RMSE : {rmse:.2f} vehicles/hr")
    print(f"R²   : {r2:.4f}")

    # 5. SHAP Global Attribution
    shap_model = ShapLSTMWrapper(eval_model)
    bg = torch.stack([train_ds[i][0] for i in np.random.choice(len(train_ds), 50, replace=False)])
    ev = torch.stack([test_ds[i][0] for i in np.random.choice(len(test_ds), 100, replace=False)])
    explainer = shap.GradientExplainer(shap_model, bg)
    shap_vals = np.squeeze(explainer.shap_values(ev))
    global_shap = np.mean(np.abs(shap_vals), axis=(0, 1)).flatten()

    df_shap = pd.DataFrame({"Feature": feature_cols, "Mean_SHAP": global_shap}).sort_values("Mean_SHAP", ascending=False)
    print("\n--- SHAP Feature Attribution Top 5 ---")
    print(df_shap.head(5).to_string(index=False))


if __name__ == "__main__":
    main()