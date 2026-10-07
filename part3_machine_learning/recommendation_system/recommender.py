"""
Part 3: Task 3 - Recommender System Engine
Provides dynamic departure window and congestion-avoidance recommendations
based on trained supervised models and historical association rules.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("traffic_recommender")


def setup_recommender_logging():
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.propagate = False


class SmartTrafficRecommender:
    """
    Intelligent departure planning and congestion-mitigation recommendation system.
    """

    def __init__(self, models_dir=None, rules_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.abspath(os.path.join(base_dir, ".."))

        self.models_dir = models_dir or os.path.join(ml_dir, "models")
        self.rules_path = rules_path or os.path.join(ml_dir, "traffic_association_rules.csv")

        self.reg_model = None
        self.clf_model = None
        self.rules_df = None

        self._load_artifacts()

    def _load_artifacts(self):
        reg_path = os.path.join(self.models_dir, "best_regression_rf.pkl")
        clf_path = os.path.join(self.models_dir, "best_classification_rf.pkl")

        logger.info(f"Loading regression model from: {reg_path}")
        self.reg_model = joblib.load(reg_path)

        logger.info(f"Loading classification model from: {clf_path}")
        self.clf_model = joblib.load(clf_path)

        if os.path.exists(self.rules_path):
            self.rules_df = pd.read_csv(self.rules_path)
            logger.info(f"Loaded {len(self.rules_df)} association rules from {self.rules_path}")
        else:
            logger.warning(f"Rules file not found at {self.rules_path}. Operating without rules.")
            self.rules_df = pd.DataFrame()

    def _build_feature_vector(self, hour, day_of_week, temp_k, rain_1h, clouds_all, weather_main):
        """Constructs an aligned DataFrame matching the trained models' feature names."""
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_holiday = 0

        low_vis = ["Fog", "Mist", "Smoke", "Haze"]
        is_low_vis = 1 if weather_main in low_vis else 0

        vector = {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "temp": temp_k,
            "rain_1h": rain_1h,
            "snow_1h": 0.0,
            "clouds_all": clouds_all,
            "is_low_visibility": is_low_vis
        }

        weather_categories = [
            "Clouds", "Drizzle", "Fog", "Haze", "Mist",
            "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"
        ]
        for w in weather_categories:
            vector[f"weather_{w}"] = 1 if weather_main == w else 0

        df_vec = pd.DataFrame([vector])

        if hasattr(self.reg_model, "feature_names_in_"):
            df_vec = df_vec[self.reg_model.feature_names_in_]

        return df_vec

    def assess_conditions(self, hour, day_of_week, temp_c=20.0, rain_1h=0.0, clouds_all=20, weather_main="Clear"):
        """Evaluates traffic volume and accident risk probability."""
        temp_k = temp_c + 273.15
        X_input = self._build_feature_vector(hour, day_of_week, temp_k, rain_1h, clouds_all, weather_main)

        pred_volume = float(self.reg_model.predict(X_input)[0])
        risk_prob = float(self.clf_model.predict_proba(X_input)[0][1])

        return {
            "hour": hour,
            "predicted_volume": round(pred_volume, 1),
            "risk_probability": round(risk_prob, 3),
            "is_high_risk": bool(risk_prob >= 0.50)
        }

    def _clean_itemset_str(self, raw_str):
        """Cleans frozenset string representations into human-readable text."""
        cleaned = str(raw_str).replace("frozenset({", "").replace("})", "").replace("'", "")
        return cleaned.strip()

    def _get_matching_rules(self, hour, day_of_week, weather_main):
        """Finds top association rule warnings that match the current scenario context."""
        if self.rules_df.empty:
            return []

        # Determine slot tags
        if 6 <= hour <= 9:
            slot = "Time_MorningRush"
        elif 10 <= hour <= 15:
            slot = "Time_Midday"
        elif 16 <= hour <= 19:
            slot = "Time_EveningRush"
        else:
            slot = "Time_NightOffPeak"

        day_tag = "Day_Weekend" if day_of_week in [5, 6] else "Day_Weekday"

        # Match rules whose antecedents contain these tags
        matching = []
        for _, row in self.rules_df.iterrows():
            antecedent_raw = str(row.get("antecedents_str", row.get("antecedents", "")))
            consequent_raw = str(row.get("consequents_str", row.get("consequents", "")))

            if slot in antecedent_raw and day_tag in antecedent_raw:
                conf = row["confidence"] * 100
                lift = row["lift"]
                ant_clean = self._clean_itemset_str(antecedent_raw)
                con_clean = self._clean_itemset_str(consequent_raw)
                matching.append(f"Historical Pattern: If [{ant_clean}] -> Expect [{con_clean}] (Confidence: {conf:.1f}%, Lift: {lift:.2f})")

        return matching[:2]

    def recommend_departure(self, target_hour, day_of_week, temp_c=20.0, rain_1h=0.0, clouds_all=20, weather_main="Clear"):
        """Evaluates adjacent windows (-2h to +2h) and returns actionable departure advice."""
        target_eval = self.assess_conditions(target_hour, day_of_week, temp_c, rain_1h, clouds_all, weather_main)

        # Candidate windows (-2 to +2 hours)
        candidate_hours = [(target_hour + delta) % 24 for delta in [-2, -1, 1, 2]]
        evaluations = [target_eval]

        for cand_h in candidate_hours:
            eval_res = self.assess_conditions(cand_h, day_of_week, temp_c, rain_1h, clouds_all, weather_main)
            eval_res["time_delta_hours"] = cand_h - target_hour
            evaluations.append(eval_res)

        # Select window with lowest volume and lowest risk
        best_candidate = min(
            evaluations,
            key=lambda x: (x["is_high_risk"], x["predicted_volume"])
        )

        volume_saved = target_eval["predicted_volume"] - best_candidate["predicted_volume"]
        pct_improvement = (volume_saved / target_eval["predicted_volume"] * 100) if target_eval["predicted_volume"] > 0 else 0.0

        if best_candidate["hour"] == target_hour or pct_improvement < 15.0:
            advice = "Current departure window is optimal. Proceed as planned."
            recommended_shift = 0
        else:
            time_diff = best_candidate["hour"] - target_hour
            direction = "earlier" if time_diff < 0 else "later"
            recommended_shift = time_diff
            advice = (
                f"Shift departure by {abs(time_diff)} hour(s) {direction} (at {best_candidate['hour']:02d}:00). "
                f"Expected traffic volume drops by {pct_improvement:.1f}% ({int(volume_saved)} fewer vehicles)."
            )

        matched_rules = self._get_matching_rules(target_hour, day_of_week, weather_main)

        return {
            "target_hour": f"{target_hour:02d}:00",
            "target_predicted_volume": target_eval["predicted_volume"],
            "target_risk_probability": f"{target_eval['risk_probability']:.1%}",
            "recommended_hour": f"{best_candidate['hour']:02d}:00",
            "recommended_volume": best_candidate["predicted_volume"],
            "recommended_risk_probability": f"{best_candidate['risk_probability']:.1%}",
            "time_shift_hours": recommended_shift,
            "improvement_pct": f"{pct_improvement:.1f}%",
            "advice": advice,
            "historical_alerts": matched_rules
        }


def run_demo():
    """Runs standard scenario evaluations demonstrating recommender capabilities."""
    setup_recommender_logging()
    logger.info("Initializing Smart Traffic Recommender Engine Demo...")
    recommender = SmartTrafficRecommender()

    scenarios = [
        {
            "name": "Scenario 1: Weekday Morning Commute in Rainy Conditions",
            "hour": 8,
            "dow": 0,  # Monday
            "temp_c": 12.0,
            "rain_1h": 3.5,
            "clouds_all": 90,
            "weather": "Rain"
        },
        {
            "name": "Scenario 2: Weekday Evening Commute in Heavy Winter Snow",
            "hour": 17,
            "dow": 3,  # Thursday
            "temp_c": -5.0,
            "rain_1h": 0.0,
            "clouds_all": 85,
            "weather": "Snow"
        },
        {
            "name": "Scenario 3: Weekend Afternoon Leisure Trip (Clear Sky)",
            "hour": 14,
            "dow": 6,  # Sunday
            "temp_c": 22.0,
            "rain_1h": 0.0,
            "clouds_all": 10,
            "weather": "Clear"
        }
    ]

    print("\n" + "=" * 95)
    print("SMART CITY TRAFFIC ADVISORY & DEPARTURE RECOMMENDER ENGINE")
    print("=" * 95)

    for sc in scenarios:
        rec = recommender.recommend_departure(
            target_hour=sc["hour"],
            day_of_week=sc["dow"],
            temp_c=sc["temp_c"],
            rain_1h=sc["rain_1h"],
            clouds_all=sc["clouds_all"],
            weather_main=sc["weather"]
        )

        print(f"\n>>> {sc['name']}")
        print(f"  Intended Departure   : {rec['target_hour']} | Est. Volume: {rec['target_predicted_volume']} | Risk: {rec['target_risk_probability']}")
        print(f"  Recommended Window   : {rec['recommended_hour']} | Est. Volume: {rec['recommended_volume']} | Risk: {rec['recommended_risk_probability']}")
        print(f"  Expected Benefit     : {rec['improvement_pct']} volume reduction ({rec['time_shift_hours']}h shift)")
        print(f"  System Advisory      : {rec['advice']}")
        if rec["historical_alerts"]:
            print("  Historical Insights  :")
            for alert in rec["historical_alerts"]:
                print(f"    * {alert}")
        print("-" * 95)


if __name__ == "__main__":
    run_demo()