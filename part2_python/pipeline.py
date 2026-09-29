# Task 1: Data Pipeline Construction (with mandatory logging)

# Import Operating system utilities, system specific functions (to route logs to stdout) 
import os
import sys
import logging # Import python's standard logging module
import pandas as pd  # Data analysis library to ingest and manage CSV data (tabular)

# Module level logger (isolated from global root logger)
logger = logging.getLogger(__name__)

# Definition of the the exact 9 base columns expected in the raw dataset
EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

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

def load_raw_data(file_path: str) -> pd.DataFrame:
    # Loads the raw traffic CSV file (with exception handling and logging at each milestone)
    # Input Parameter : file_path (str) - Path to the CSV file
    # Output : Dataframe loaded with the csv file
    
    # Check if the file exists before attempting to open or load it
    if not os.path.exists(file_path):
        logger.error(f"Input file not found at path: {file_path}", exc_info=True)
        raise FileNotFoundError(f"Missing input dataset: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        # Log an INFO message when the raw file is successfully loaded with number of rows and columns
        logger.info(
            f"CSV loaded successfully from '{file_path}'. "
            f"Number of rows and columns: {df.shape[0]} rows and {df.shape[1]} columns."
        )
        return df
    except pd.errors.EmptyDataError as ede:
        # Logs error if the file is empty
        logger.error(f"File is completely empty: {ede}", exc_info=True)
        raise

    except pd.errors.ParserError as pe:
        # Logs error if the csv file format is not correct or corrupted
        logger.error(f"CSV Parsing failure in '{file_path}': {pe}", exc_info=True)
        raise

    except Exception as e:
        # Logs all other standard system and file I/O issues
        logger.error(f"Unexpected error loading '{file_path}': {e}", exc_info=True)
        raise

# Validates the data before performing any other data processing operations. 
# Checks that all expected columns are present in the dataset.
# Input is the dataframe loaded from the csv (pd.DataFrame)
# Output indicates if all required columns are present, if doesnt it raises a ValueError
def validate_schema(df: pd.DataFrame) -> bool:
    # Identify which of the base columns are missing
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    # If any expected columns are missing, log an ERROR and halt execution
    if missing_columns:
        logger.error(
            f"Schema failed validation. Expected columns missing: {missing_columns}"
        )
        raise ValueError(f"Schema mismatch: columns missing {missing_columns}")

    # Logging after all columns are verified
    logger.info(
        f"Schema validation passed: all {len(EXPECTED_COLUMNS)} expected columns are present."
    )
    return True


# Clean date/time fields, duplicates, and standardise the text strings
# It also logs each clean operation separately with the number of rows and the reason.
# Input is the dataframe loaded from the csv (pd.DataFrame)
# Output is the dataframe with the dates parsed, after removing duplicates, and removing uniform text formatting.
def clean_basic_data(df: pd.DataFrame) -> pd.DataFrame:

    cleaned = df.copy()
    cleaned["date_time"] = pd.to_datetime(cleaned["date_time"], errors="coerce")
    invalid_dates_count = cleaned["date_time"].isna().sum()
    
    if invalid_dates_count > 0:
        # Requirement: Show WARNING before removing rows with reason and count
        logger.warning(
            f"{invalid_dates_count} row(s) dropped as unable to parse formatting of date_time."
        )
        cleaned = cleaned.dropna(subset=["date_time"])
    else:
        logger.info("Date/time validation passed: 0 invalid date stamps detected.")

    # Identification and removal of duplicated rows
    duplicate_rows_count = cleaned.duplicated().sum()
    
    if duplicate_rows_count > 0:
        # Assignment Requirement: Show WARNING before removing rows with reason and count 
        cleaned = cleaned.drop_duplicates()
        logger.warning(
            f"{duplicate_rows_count} row(s) dropped as unable to ensure uniqueness of data"
        )
    else:
        logger.info("Duplicate row check passed and number of duplicate records found is 0")

    # Code section standardizes inconsisent values in the category related fields
    # Will strip whitepace and case normalisation on all category columns in string format
    cleaned["holiday"] = cleaned["holiday"].astype(str).str.strip()
    cleaned["weather_main"] = cleaned["weather_main"].astype(str).str.strip().str.title()
    cleaned["weather_description"] = (
        cleaned["weather_description"].astype(str).str.strip().str.lower()
    )
    
    logger.info(
        "Fields with type related to category are standardized ('holiday', 'weather_main', 'weather_description'): "
        "whitespace have been removed or stripped and case has been normalised"
    )

    return cleaned

# Detect and handle outliers or values which are impossible (0 Kelvin temp and extreme rainfall)
# Outliers are imputed using group-by-group monthly median using loops and conditionals
# Input parameter : df (pd.DataFrame) - Dataframe cleaned with parsed dates
# Output : df (pd.DataFrame) - Dataframe with outliers imputed using monthly median
def impute_outliers_monthly(df: pd.DataFrame) -> pd.DataFrame:
    imputed_df = df.copy()

    imputed_df["_month"] = imputed_df["date_time"].dt.month

    # Handling outliers of temperature 
    temp_outlier_mask = imputed_df["temp"] <= 0.0
    temp_outliers_count = temp_outlier_mask.sum()

    if temp_outliers_count > 0:
        logger.warning(
            f" temperature outlier {temp_outliers_count} identified row(s) (temp <= 0 k) "
            f"Imputing values using group-by-group monthly median."
        )
        for month_val in sorted(imputed_df["_month"].unique()):
            # Calculate median excluding unphysical zero values for this month
            valid_month_temps = imputed_df.loc[
                (imputed_df["_month"] == month_val) & (~temp_outlier_mask), "temp"
            ]
            monthly_median_temp = valid_month_temps.median()

            month_target = (imputed_df["_month"] == month_val) & temp_outlier_mask
            rows_to_impute = month_target.sum()

            if rows_to_impute > 0:
                imputed_df.loc[month_target, "temp"] = monthly_median_temp
                logger.info(
                    f"Month {month_val}: Imputed {rows_to_impute} temperature outlier(s) "
                    f"with monthly median of {monthly_median_temp:.2f} K."
                )
    else:
        logger.info("Temperature outlier check passed: 0 impossible readings found.")

    # Rainfall Outlier Handling (> 500 mm/h physically impossible sensor spikes)
    rain_outlier_mask = imputed_df["rain_1h"] > 500.0
    rain_outliers_count = rain_outlier_mask.sum()

    if rain_outliers_count > 0:
        logger.warning(
            f"Identified {rain_outliers_count} rainfall outlier row(s) (> 500 mm/h). "
            f"Imputing values using group-by-group monthly median."
        )
        for month_val in sorted(imputed_df["_month"].unique()):
            valid_month_rain = imputed_df.loc[
                (imputed_df["_month"] == month_val) & (~rain_outlier_mask), "rain_1h"
            ]
            monthly_median_rain = valid_month_rain.median()

            month_target = (imputed_df["_month"] == month_val) & rain_outlier_mask
            rows_to_impute = month_target.sum()

            if rows_to_impute > 0:
                imputed_df.loc[month_target, "rain_1h"] = monthly_median_rain
                logger.info(
                    f"Month {month_val}: Imputed {rows_to_impute} rainfall outlier(s) "
                    f"with monthly median of {monthly_median_rain:.2f} mm."
                )
    else:
        logger.info("Rainfall outlier check passed: 0 impossible readings found.")

    imputed_df = imputed_df.drop(columns=["_month"])
    return imputed_df

# Create new columns from date and weather data
def add_new_features(data: pd.DataFrame) -> pd.DataFrame:
    df_with_features = data.copy()

    # Extract parts from date and time
    df_with_features["hour"] = df_with_features["date_time"].dt.hour
    df_with_features["day_of_week"] = df_with_features["date_time"].dt.dayofweek
    df_with_features["month"] = df_with_features["date_time"].dt.month

    # Check for the day, if it is a weekend (5 is Saturday, 6 is Sunday)
    is_saturday_or_sunday = df_with_features["day_of_week"] >= 5
    df_with_features["is_weekend"] = is_saturday_or_sunday.astype(int)

    # Check for peak/rush hour on weekdays (7-9 AM and 4-6 PM)
    is_weekday = df_with_features["is_weekend"] == 0
    is_morning_rush = (df_with_features["hour"] >= 7) & (df_with_features["hour"] <= 9)
    is_evening_rush = (df_with_features["hour"] >= 16) & (df_with_features["hour"] <= 18)
    
    is_rush_hour = is_weekday & (is_morning_rush | is_evening_rush)
    df_with_features["rush_hour"] = is_rush_hour.astype(int)

    # convert temperature from Kelvin to Celsius
    celsius_temperature = df_with_features["temp"] - 273.15
    df_with_features["temp_celsius"] = celsius_temperature.round(2)

    # adverse weather check (rain, snow, thunderstorms, mist, fog, haze, squalls)
    adverse_weather_list = ["Rain", "Snow", "Thunderstorm", "Mist", "Fog", "Haze", "Squall", "Smoke"]
    df_with_features["is_adverse_weather"] = (
        df_with_features["weather_main"].isin(adverse_weather_list).astype(int)
    )

    logger.info(
        f"Feature engineering completed - added 7 new columns "
        f"(hour, day_of_week, month, is_weekend, rush_hour, temp_celsius, is_adverse_weather). "
        f"Total columns : {df_with_features.shape[1]}."
    )
    return df_with_features


if __name__ == "__main__":
    setup_logging()

    # Path to the  CSV located the project root
    raw_data_path = "Metro_Interstate_Traffic_Volume.csv"
    
    try:
        # Load the data from csv 
        raw_df = load_raw_data(raw_data_path)
        
        # Validate to ensure all the columns are present before performing further processing
        validate_schema(raw_df)
        
        # Parsing of dates, duplicates will be removed and text's standarsed
        cleaned_df = clean_basic_data(raw_df)
        
        # Step to detect outliers and impute using monthly medians
        final_df = impute_outliers_monthly(cleaned_df)
        
        # Create new features
        features = add_new_features(final_df)
    
        # save the process file to a csv file for verification
        features.to_csv("part2_python/traffic_features.csv", index=False)
        logger.info("dataset processed with features has been saved to 'part2_python/traffic_features.csv'.")
    except Exception as err:
        logger.error(f"Loading failed during load: {err}", exc_info=True)
        sys.exit(1)