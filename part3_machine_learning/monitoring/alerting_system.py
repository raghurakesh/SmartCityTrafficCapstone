"""
Part 3 - Task 6.5: Alerting Mechanism & Operational Health Dashboard
Consolidates drift detection results into an executive health check:
  - PASS / Normal
  - ALERT / Requires investigation
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

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

# Import monitoring baseline and window simulator
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from monitor_drift import load_data_and_model, simulate_production_windows
from scipy.stats import ks_2samp


class ProductionAlertSystem:
    def __init__(self, p_threshold=0.01, ks_alert_threshold=0.25):
        self.p_threshold = p_threshold
        self.ks_alert_threshold = ks_alert_threshold

    def evaluate_health(self, window_name, df_base, df_prod):
        eval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        monitored = [
            ("temp_c", "Covariate: Temperature (°C)"),
            ("clouds_all", "Covariate: Cloud Cover (%)"),
            ("traffic_volume", "Concept / Prediction Volume")
        ]

        flags = []
        table_rows = []

        for col, desc in monitored:
            base_vals = df_base[col].dropna().values
            prod_vals = df_prod[col].dropna().values

            ks_stat, p_val = ks_2samp(base_vals, prod_vals)
            drift_flag = (p_val < self.p_threshold) and (ks_stat > self.ks_alert_threshold)

            if drift_flag:
                flags.append((desc, ks_stat, p_val))
                row_status = "CRITICAL DRIFT"
            else:
                row_status = "HEALTHY"

            table_rows.append({
                "Telemetry Entity": desc,
                "Reference Mean": f"{np.mean(base_vals):.2f}",
                "Batch Mean": f"{np.mean(prod_vals):.2f}",
                "KS Statistic": f"{ks_stat:.4f}",
                "p-value": f"{p_val:.2e}",
                "Metric Status": row_status
            })

        # Overall 6.5 System Status
        system_status = "ALERT / Requires investigation" if len(flags) > 0 else "PASS / Normal"

        logger.info(f"SMART CITY TRAFFIC MLOps ALERT DASHBOARD - {eval_time}")
        logger.info(f"Monitoring scope: {window_name}")
        logger.info(f"System state: [{system_status}]")

        for row in table_rows:
            log_row = logger.warning if row["Metric Status"] == "CRITICAL DRIFT" else logger.info
            log_row(
                f"{row['Telemetry Entity']} | reference mean: {row['Reference Mean']}, "
                f"batch mean: {row['Batch Mean']}, KS: {row['KS Statistic']}, "
                f"p-value: {row['p-value']} -> {row['Metric Status']}"
            )

        if system_status == "PASS / Normal":
            logger.info("Action protocol: [AUTOMATED SIGN-OFF] - all live features align with baseline training distributions.")
        else:
            logger.warning(f"Action protocol: [ESCALATION TRIGGERED] - {len(flags)} signal(s) tripped:")
            for name, ks, p in flags:
                logger.warning(f"  * {name} -> KS: {ks:.4f}, p: {p:.2e}")
            logger.warning("  Mitigation 1: route notification to Municipal Operations Engineering channel.")
            logger.warning("  Mitigation 2: quarantine affected window from continuous automated feedback loop.")
            logger.warning("  Mitigation 3: trigger automated model retraining on newly acquired weather regime.")

        return system_status


if __name__ == "__main__":
    setup_logging()
    df_base, _ = load_data_and_model()
    df_win1, df_win2 = simulate_production_windows(df_base)

    alert_engine = ProductionAlertSystem(p_threshold=0.01, ks_alert_threshold=0.25)

    # 1. Test nominal window
    alert_engine.evaluate_health("Batch Run: Weekday Standard Operations", df_base, df_win1)

    # 2. Test anomalous drifted window
    alert_engine.evaluate_health("Batch Run: Arctic Blizzard Influx", df_base, df_win2)