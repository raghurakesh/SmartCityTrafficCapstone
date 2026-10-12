"""
Part 3 - Task 6.3: Model Deployment Mock-up
FastAPI service serving the champion Random Forest Traffic Volume model.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

# Standard logging configuration matching Part 2
logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "..", "ml_pipeline.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


setup_logging()

app = FastAPI(
    title="Smart City Traffic Volume Prediction API",
    description="Task 6.3 deployment mock-up serving champion model.",
    version="1.0.0"
)

# Load Champion Model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "best_regression_rf.pkl"))

if not os.path.exists(model_path):
    logger.error(f"Model file not found at: {model_path}")
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = joblib.load(model_path)
logger.info(f"Champion model loaded from '{model_path}'")

class TrafficInput(BaseModel):
    hour: int = Field(8, ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(0, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    temp_c: float = Field(20.0, description="Temperature in Celsius")
    rain_1h: float = Field(0.0, ge=0.0, description="Rainfall in last 1 hour (mm)")
    clouds_all: int = Field(20, ge=0, le=100, description="Cloud cover percentage (0-100)")
    weather_main: str = Field("Clear", description="Dominant weather condition")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": "v1.1-RandomForestRegressor",
        "features_count": len(model.feature_names_in_) if hasattr(model, "feature_names_in_") else 21
    }


@app.post("/predict")
def predict_traffic(payload: TrafficInput):
    try:
        temp_k = payload.temp_c + 273.15
        hour_sin = np.sin(2 * np.pi * payload.hour / 24.0)
        hour_cos = np.cos(2 * np.pi * payload.hour / 24.0)
        dow_sin = np.sin(2 * np.pi * payload.day_of_week / 7.0)
        dow_cos = np.cos(2 * np.pi * payload.day_of_week / 7.0)
        is_weekend = 1 if payload.day_of_week in [5, 6] else 0
        is_holiday = 0
        is_low_vis = 1 if payload.weather_main in ["Fog", "Mist", "Smoke", "Haze"] else 0

        vector = {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "temp": temp_k,
            "rain_1h": payload.rain_1h,
            "snow_1h": 0.0,
            "clouds_all": payload.clouds_all,
            "is_low_visibility": is_low_vis
        }

        weather_categories = [
            "Clouds", "Drizzle", "Fog", "Haze", "Mist",
            "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"
        ]
        for w in weather_categories:
            vector[f"weather_{w}"] = 1 if payload.weather_main == w else 0

        df_input = pd.DataFrame([vector])
        if hasattr(model, "feature_names_in_"):
            df_input = df_input[model.feature_names_in_]

        prediction = float(model.predict(df_input)[0])
        logger.info(
            f"Prediction served: hour={payload.hour}, day_of_week={payload.day_of_week}, "
            f"weather={payload.weather_main}, predicted_volume={prediction:.1f}"
        )

        return {
            "status": "success",
            "model_version": "v1.1-RandomForestRegressor",
            "hour": payload.hour,
            "predicted_traffic_volume": round(prediction, 1),
            "unit": "vehicles/hour"
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))