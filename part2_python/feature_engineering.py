# Task 2: Feature Engineering (NumPy and Pandas)

# Import Operating system utilities, system specific functions (to route logs to stdout) 
import os
import sys
import logging # Import python's standard logging module
import pandas as pd  # Data analysis library to ingest and manage CSV data (tabular)
import numpy as np

# Module level logger (isolated from global root logger)
logger = logging.getLogger(__name__)

"""
Configures logging in console and file with structured formatting as per the requirement
 - No root logger used directly
 - StreamHandler for console and FileHandler for disk
 - Standardised format with Timestamp, Level, Module and Message
 - No print() statement used for status/progress reporting or logging
"""

def setup_logging(log_file: str = "part2_python/pipeline.log", level=logging.INFO):
    log_dir = os.path.dirname(log_file)
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Handler to print log messages directly to the terminal and console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Handler to write log messages into a physical file on disk (mode 'w' overwrites the file on a fresh run)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
def load_feature_data(file_path: str) -> pd.DataFrame:
    # Loads the processed traffic CSV file (with exception handling and logging at each milestone)
    # Input Parameter : file_path - Path to the CSV file
    # Output : Dataframe loaded with the csv file
    
    if not os.path.exists(file_path):
        logger.error(f"Input file not found at path: {file_path}", exc_info=True)
        raise FileNotFoundError(f"Missing input dataset: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(
            f"Shape before feature engineering: "
            f"{df.shape[0]} rows and {df.shape[1]} columns."
        )
        return df
    except Exception as e:
        logger.error(f"Unexpected error loading '{file_path}': {e}", exc_info=True)
        raise
    
def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert 24-hour time into circular coordinates (sine and cosine)
    # Input parameter : df (pd.DataFrame)
    # Output : df (pd.DataFrame) with hour_sin and hour_cos columns added
    
    df_cyclical = df.copy()

    # Step A: Fraction of the day that has passed (0.0 to 1.0)
    fraction_of_day = df_cyclical["hour"] / 24.0

    # Step B: Full circle angle in radians (360 degrees = 2 * pi)
    circle_angle = 2.0 * np.pi * fraction_of_day

    # Step C: Coordinates on the circle
    df_cyclical["hour_sin"] = np.sin(circle_angle).round(4)
    df_cyclical["hour_cos"] = np.cos(circle_angle).round(4)

    logger.info(
        "Cyclical time features added: 'hour_sin' and 'hour_cos'. "
        f"Total columns is now {df_cyclical.shape[1]}."
    )
    return df_cyclical

# Create normalised/scaled versions of continuous variables into 0 to 1 range
# Input parameter : df (pd.DataFrame)
# Output : df (pd.DataFrame) with scaled numerical columns added
def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    df_scaled = df.copy()

    # Continuous variables to scale
    columns_to_scale = ["temp_celsius", "traffic_volume"]

    for col in columns_to_scale:
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        
        # Min-Max Normalisation formula: (X - min) / (max - min)
        scaled_column_name = f"{col}_scaled"
        df_scaled[scaled_column_name] = (
            (df_scaled[col] - min_val) / (max_val - min_val)
        ).round(4)

    logger.info(
        "Normalised continuous variables added: 'temp_celsius_scaled', 'traffic_volume_scaled'. "
        f"Total columns is now {df_scaled.shape[1]}."
    )
    return df_scaled

# Create data-driven congestion category based on traffic volume quartiles
# Documents logic and logs threshold values at DEBUG level as required
# Input parameter : df (pd.DataFrame)
# Output : df (pd.DataFrame) with congestion_category column added
def add_congestion_category(df: pd.DataFrame) -> pd.DataFrame:
    df_congestion = df.copy()

    # Calculate quartile thresholds (25%, 50%, 75%)
    q1 = df_congestion["traffic_volume"].quantile(0.25)
    q2 = df_congestion["traffic_volume"].quantile(0.50)
    q3 = df_congestion["traffic_volume"].quantile(0.75)

    # Requirement: Log intermediate values/thresholds at the DEBUG level
    logger.debug(
        f"Congestion quartile thresholds calculated: "
        f"Q1 (25%) = {q1:.2f}, Q2 (50%) = {q2:.2f}, Q3 (75%) = {q3:.2f}"
    )

    # Bucketing function based on quartiles
    def categorize_traffic(volume: float) -> str:
        if volume <= q1:
            return "Low"
        elif volume <= q2:
            return "Medium"
        elif volume <= q3:
            return "High"
        else:
            return "Severe"

    df_congestion["congestion_category"] = df_congestion["traffic_volume"].apply(categorize_traffic)

    logger.info(
        "Data-driven 'congestion_category' added based on quartile distribution (Low, Medium, High, Severe). "
        f"Total columns is now {df_congestion.shape[1]}."
    )
    return df_congestion

if __name__ == "__main__":
    setup_logging()

    input_data_path = "part2_python/traffic_features.csv"
    
    try:
        traffic_df = load_feature_data(input_data_path)

        # Step: Add cyclical hour features
        traffic_df = add_cyclical_time_features(traffic_df)
    
        # Step: Add normalised continuous features
        traffic_df = add_scaled_features(traffic_df)

        # Step: Add data-driven congestion categories
        traffic_df = add_congestion_category(traffic_df)
#
#       Requirement: Log the dataset shape after feature engineering
        logger.info(
            f"Shape after feature engineering: "
            f"{traffic_df.shape[0]} rows and {traffic_df.shape[1]} columns."
        )

        # Save processed dataset back to traffic_features.csv for downstream tasks
        output_file_path = "part2_python/traffic_features.csv"
        traffic_df.to_csv(output_file_path, index=False)
        logger.info(f"Enriched feature dataset successfully saved to '{output_file_path}'.")
        
    except Exception as err:
        logger.error(f"Loading failed during load: {err}", exc_info=True)
        sys.exit(1)