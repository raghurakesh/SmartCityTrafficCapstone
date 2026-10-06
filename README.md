# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

AI, ML, and Data Science Capstone Project analyzing westbound Interstate 94 traffic volume and weather data.

## Project Structure

- `PART1DataAnalytics/`: SQL queries, statistical analysis, and Power BI assets.
- `PART2Python`: Data pipeline, feature engineering, visualizations, and CLI app.
- `PART3MachineLearning/`: Machine learning models, deep learning, MLOps, and governance reports.

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
├── part1_data_analytics/                # SQL queries and statistical analysis
└── part2_python/
    ├── pipeline.log                     # Standardized execution and audit log
    ├── traffic_features.csv             # Enriched dataset with engineered features
    ├── feature_engineering.py           # Feature engineering transformations
    ├── visualizations.py                # Visual analytics script
    ├── figures/                         # Generated high-resolution plots
    │   ├── 01_hourly_traffic_weekday_vs_weekend.png
    │   ├── 02_traffic_by_weather_condition.png
    │   └── 03_traffic_by_congestion_category.png
    └── cli_app/
        └── traffic_app.py               # Interactive CLI analytics application
```

---

## Installation and setup

### 1. Clone the repository

```bash
git clone [https://github.com/raghurakesh/SmartCityTrafficCapstone.git](https://github.com/raghurakesh/SmartCityTrafficCapstone.git)
cd SmartCityTrafficCapstone
```

### 2. Setting up the Python environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas matplotlib seaborn
```

---

## Execution Guide

### 1. Feature Engineering

Runs cyclical time encoding, Min-Max feature scaling, and quartile-based congestion bucketing:

```bash
python part2_python/feature_engineering.py
```

**Output:** Generates `part2_python/traffic_features.csv`

### 2. Visualisation and Charts

Generates high-resolution Matplotlib and Seaborn figures:

```bash
python part2_python/visualizations.py
```

**Output:** Saves 3 figures under `part2_python/figures/`

- `01_hourly_traffic_weekday_vs_weekend.png`
- `02_traffic_by_weather_condition.png`
- `03_traffic_by_congestion_category.png`

### 3. Interactive analytics application in CLI

Launches the command-line analytics application:

```bash
python part2_python/cli_app/traffic_app.py
```

#### Capabilities of the application

- View overall traffic summary statistics (mean, std, quartiles, min, max).
- Filter traffic summary metrics by specific weather conditions.
- Compare peak Rush Hour vs. Non-Rush Hour traffic distributions.
- Built-in input validation and user audit logging.

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
