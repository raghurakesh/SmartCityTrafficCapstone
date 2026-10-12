"""
Part 3 - Task 6.5: Alerting Mechanism & Operational Health Dashboard
Consolidates drift detection results into an executive health check:
  - PASS / Normal
  - ALERT / Requires investigation
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

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

        print("=" * 86)
        print(f"  SMART CITY TRAFFIC MLOps ALERT DASHBOARD - {eval_time}")
        print("=" * 86)
        print(f"Monitoring Scope : {window_name}")
        print(f"System State     : [{system_status}]")
        print("-" * 86)

        df_display = pd.DataFrame(table_rows)
        print(df_display.to_string(index=False))
        print("-" * 86)

        if system_status == "PASS / Normal":
            print("Action Protocol  : [AUTOMATED SIGN-OFF]")
            print("  - All live features align with baseline training distributions.")
            print("  - Automated inference served without intervention.")
        else:
            print("Action Protocol  : [ESCALATION TRIGGERED]")
            print(f"  - Tripped Sensors/Signals ({len(flags)}):")
            for name, ks, p in flags:
                print(f"    * {name} -> KS: {ks:.4f}, p: {p:.2e}")
            print("  - Mitigation Playbook:")
            print("    1. Route notification to Municipal Operations Engineering channel.")
            print("    2. Quarantine affected window from continuous automated feedback loop.")
            print("    3. Trigger automated model retraining on newly acquired weather regime.")
        print("=" * 86 + "\n")

        return system_status


if __name__ == "__main__":
    df_base, _ = load_data_and_model()
    df_win1, df_win2 = simulate_production_windows(df_base)

    alert_engine = ProductionAlertSystem(p_threshold=0.01, ks_alert_threshold=0.25)

    # 1. Test nominal window
    alert_engine.evaluate_health("Batch Run: Weekday Standard Operations", df_base, df_win1)

    # 2. Test anomalous drifted window
    alert_engine.evaluate_health("Batch Run: Arctic Blizzard Influx", df_base, df_win2)