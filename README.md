# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

AI, ML and Data Science Capstone Project analysing westbound Interstate 94 traffic volume and weather data (Metro Interstate Traffic Volume dataset, ~48,000 hourly records).

## Project Structure

- `part1_data_analytics/`: SQL queries, statistical and probability analysis, Power BI dashboard.
- `part2_python/`: Data pipeline, feature engineering, visualisations, CLI app, logging.
- `part3_machine_learning/`: Supervised and unsupervised models, LSTM with SHAP, MLflow tracking, recommender, deployment, monitoring, responsible AI report.

---

# Smart City Traffic Analytics System

A modular Python-based data engineering and analytics solution for analyzing urban highway traffic demand, feature transformation, visual analytics, and interactive command-line data exploration.

---

## Repository Structure

```text
SmartCityTrafficCapstone/
├── Metro_Interstate_Traffic_Volume.csv  # Raw traffic dataset
├── README.md                            # Project documentation
├── traffic.db                           # SQLite database (Part 1)
├── part1_data_analytics/
│   ├── Task-1/                          # SQL trend and holiday analysis write-ups
│   ├── sql/traffic_analysis.sql         # SQL queries
│   ├── statistics/                      # Descriptive stats, probability scripts and write-ups
│   └── dashboard/                       # Power BI .pbix, screenshot, Task-4 analysis
├── part2_python/
│   ├── pipeline.py                      # Load, validate, clean, impute
│   ├── feature_engineering.py           # Cyclical, scaled, congestion features
│   ├── visualizations.py                # Matplotlib figures
│   ├── cli_app/traffic_app.py           # Command-line analytics app
│   ├── figures/                         # Generated plots
│   ├── pipeline.log                     # Sample log output
│   └── Smart City Traffic Analytics Project Report.pdf
└── part3_machine_learning/
    ├── train_supervised.py              # Regression + classification models
    ├── train_unsupervised.py            # K-means + association rules
    ├── train_deep_learning.py           # LSTM + SHAP
    ├── track_experiments_mlflow.py      # MLflow experiment tracking
    ├── mlflow.db, mlflow_logs/          # MLflow database and CSV exports
    ├── recommendation_system/           # Travel-window recommender
    ├── deployment/app.py                # FastAPI mock-up
    ├── monitoring/                      # Drift monitoring and alerting
    ├── notebooks/                       # Three Jupyter notebooks
    ├── models/                          # Saved models
    └── *.pdf                            # Task documentation and bias/fairness report
```

---

## Installation and setup

### 1. Clone the repository

```bash
git clone https://github.com/raghurakesh/SmartCityTrafficCapstone.git
cd SmartCityTrafficCapstone
```

### 2. Set up the Python environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn scipy mlxtend torch shap mlflow joblib fastapi uvicorn pydantic
```

All commands below are run from the repository root.

---

## Execution Guide

### Part 1: Data Analytics

```bash
# SQL queries (run against traffic.db)
sqlite3 traffic.db < part1_data_analytics/sql/traffic_analysis.sql

# Descriptive statistics and correlation (Task 2)
python part1_data_analytics/statistics/descriptive_stats.py

# Probability analysis (Task 3)
python part1_data_analytics/statistics/probability_analysis.py
```

The Power BI dashboard is `part1_data_analytics/dashboard/traffic_dashboard.pbix` (screenshot: `dashboard_view.png`). Written analysis for each task is in the `.md` files inside `Task-1/`, `statistics/` and `dashboard/`.

### Part 2: Python Pipeline

#### 1. Data Pipeline

Loads the raw CSV, validates the schema, cleans the data and imputes outliers with monthly medians:

```bash
python part2_python/pipeline.py
```

**Output:** Generates `part2_python/traffic_features.csv`

#### 2. Feature Engineering

Runs cyclical time encoding, Min-Max feature scaling, and quartile-based congestion bucketing:

```bash
python part2_python/feature_engineering.py
```

**Output:** Generates `part2_python/traffic_features.csv`

#### 3. Visualisation and Charts

Generates high-resolution Matplotlib and Seaborn figures:

```bash
python part2_python/visualizations.py
```

**Output:** Saves 3 figures under `part2_python/figures/`

- `01_hourly_traffic_weekday_vs_weekend.png`
- `02_traffic_by_weather_condition.png`
- `03_traffic_by_congestion_category.png`

#### 4. Interactive analytics application in CLI

Launches the command-line analytics application:

```bash
python part2_python/cli_app/traffic_app.py
```

#### Capabilities of the application

- View overall traffic summary statistics (mean, std, quartiles, min, max).
- Filter traffic summary metrics by specific weather conditions.
- Compare peak Rush Hour vs. Non-Rush Hour traffic distributions.
- Built-in input validation and user audit logging.

### Part 3: Machine Learning

Run these after `pipeline.py` and `feature_engineering.py` have produced `part2_python/traffic_features.csv`.

```bash
# Task 1: supervised models (regression + classification), saves models/*.pkl
python part3_machine_learning/train_supervised.py

# Task 2: K-means clustering and association rules
python part3_machine_learning/train_unsupervised.py

# Task 3: LSTM with SHAP explainability
python part3_machine_learning/train_deep_learning.py

# Task 4 / 6.2: MLflow experiment tracking (writes part3_machine_learning/mlflow.db)
python part3_machine_learning/track_experiments_mlflow.py

# Task 5: travel-window recommendation demo
python part3_machine_learning/recommendation_system/recommender.py

# Task 6.4 / 6.5: drift monitoring and PASS/ALERT report
python part3_machine_learning/monitoring/alerting_system.py
```

**View MLflow results:**

```bash
mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow.db --port 5001
```

Then open http://localhost:5001. Readable exports of the runs are in `part3_machine_learning/mlflow_logs/`. The large `mlruns/` model-artifact folder is not committed (about 800 MB); re-run the tracking script to regenerate it.

**Serve the model (Task 6.3):**

```bash
uvicorn part3_machine_learning.deployment.app:app --port 8000
```

Then open http://localhost:8000/docs to try `POST /predict`, or check `GET /health`.

The Jupyter notebooks in `part3_machine_learning/notebooks/` cover the same work interactively.

---

## Accident Data: Documented Proxy (No Real Accident Dataset)

**No real accident dataset was sourced for this project.** The classification task uses a documented proxy accident-risk label, as specified in the capstone brief. It demonstrates the classification workflow and must not be read as a prediction of actual accidents.

- `congestion_category` is built from the quartiles of `traffic_volume` (Low, Medium, High, Severe).
- `high_risk = 1` when congestion is High or Severe **and** the weather is severe or low-visibility (fog, mist, haze, smoke and similar conditions).
- The label is implemented in `part3_machine_learning/train_supervised.py`.

---

## Documentation and Results

Results, interpretation and methodology are documented in the reports rather than in this README:

| Topic                                                                   | Location                                                                   |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Part 1 analysis (SQL, statistics, probability, dashboard)               | `.md` files in `part1_data_analytics/Task-1/`, `statistics/`, `dashboard/` |
| Part 2 methodology and findings                                         | `part2_python/Smart City Traffic Analytics Project Report.pdf`             |
| Part 3 Tasks 1-5 (models, clustering, LSTM + SHAP, MLflow, recommender) | `part3_machine_learning/Part3_Task1-5_Documentation.pdf`                   |
| Part 3 Task 6 (versioning, tracking, deployment, monitoring, alerting)  | `part3_machine_learning/Part3_Task6_Documentation.pdf`                     |
| Part 3 Task 7 (bias, fairness, governance, sustainability)              | `part3_machine_learning/responsible_ai_report.pdf`                         |
| MLflow run logs                                                         | `part3_machine_learning/mlflow_logs/`                                      |

---

## Logging Framework

All modules utilize Python's standard logging library configured with dual handlers:

- **Console (StreamHandler):** Real-time operational feedback.
- **File (FileHandler):** Persistent records appended to `part2_python/pipeline.log`.
- **Standard Format:** `%(asctime)s - %(levelname)s - %(name)s - %(message)s`

### Log Levels Employed:

- **DEBUG:** Per-feature row-level transformation audits.
- **INFO:** Module milestones, output file paths, CLI user actions.
- **WARNING:** Handled user input anomalies.
- **ERROR:** File load or execution failures with tracebacks.
