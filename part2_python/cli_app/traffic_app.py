# Task 4: Mini Traffic Analytics CLI Application

import os
import sys
import logging
import pandas as pd

# Module level logger
logger = logging.getLogger(__name__)

"""
Configures logging in console and file with structured formatting as per the requirement
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

    # Handler to write log messages into physical file on disk (append mode)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_dataset(file_path: str = "part2_python/traffic_features.csv") -> pd.DataFrame:
    # Loads the enriched traffic CSV dataset with validation
    # Input Parameter : file_path (str)
    # Output : pd.DataFrame
    if not os.path.exists(file_path):
        logger.error(f"Input file not found at path: {file_path}", exc_info=True)
        raise FileNotFoundError(f"Missing input dataset: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"CLI Application loaded dataset with {df.shape[0]} records and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}", exc_info=True)
        raise


def display_menu():
    # Displays the main application options to the user in the console
    print("\n==================================================")
    print("      SMART CITY TRAFFIC ANALYTICS SYSTEM        ")
    print("==================================================")
    print("1. View Overall Traffic Summary Statistics")
    print("2. Filter Traffic by Weather Condition")
    print("3. Filter Traffic by Rush Hour Status")
    print("4. Exit Application")
    print("==================================================")

def display_traffic_summary(series: pd.Series, label: str = "Overall Dataset"):
    # Computes and displays descriptive statistics for traffic volume
    # Input parameters:
    #   series (pd.Series): The traffic volume column to analyze
    #   label (str): Title/label for this particular slice of data
    
    total_count = len(series)
    mean_val = series.mean()
    std_val = series.std()
    min_val = series.min()
    q1_val = series.quantile(0.25)
    median_val = series.median()
    q3_val = series.quantile(0.75)
    max_val = series.max()

    print(f"\n==================================================")
    print(f"       TRAFFIC SUMMARY: {label.upper()}")
    print(f"==================================================")
    print(f"Total Observations : {total_count:,}")
    print(f"Mean Volume        : {mean_val:.2f} vehicles/hour")
    print(f"Standard Deviation : {std_val:.2f}")
    print(f"Minimum Volume     : {min_val:,} vehicles/hour")
    print(f"25th Percentile    : {q1_val:.2f} vehicles/hour")
    print(f"Median (50th)      : {median_val:.2f} vehicles/hour")
    print(f"75th Percentile    : {q3_val:.2f} vehicles/hour")
    print(f"Maximum Volume     : {max_val:,} vehicles/hour")
    print(f"==================================================")

def handle_weather_filter(df: pd.DataFrame):
    # Dynamically lists distinct weather conditions, prompts user, and displays filtered summary
    # Input parameter: df (pd.DataFrame)
    weather_options = sorted(df["weather_main"].dropna().unique().tolist())

    print("\n--------------------------------------------------")
    print("           SELECT WEATHER CONDITION               ")
    print("--------------------------------------------------")
    for idx, weather in enumerate(weather_options, start=1):
        print(f"  {idx}. {weather}")
    print("--------------------------------------------------")

    choice = input(f"Enter weather number (1-{len(weather_options)}): ").strip()

    if not choice.isdigit() or not (1 <= int(choice) <= len(weather_options)):
        print("\n[!] Invalid weather selection. Returning to main menu.")
        logger.warning(f"Invalid weather choice entered: '{choice}'.")
        return

    selected_weather = weather_options[int(choice) - 1]
    filtered_df = df[df["weather_main"] == selected_weather]

    logger.info(
        f"Filtered dataset by weather '{selected_weather}': "
        f"{len(filtered_df)} matches found."
    )

    display_traffic_summary(
        filtered_df["traffic_volume"],
        label=f"Weather: {selected_weather}"
    )

def handle_rush_hour_filter(df: pd.DataFrame):
    # Prompts user to select rush hour or non-rush hour and displays summary stats
    # Input parameter: df (pd.DataFrame)
    print("\n--------------------------------------------------")
    print("           SELECT RUSH HOUR STATUS                ")
    print("--------------------------------------------------")
    print("  1. Rush Hour (Weekday 07:00-09:00 & 16:00-18:00)")
    print("  2. Non-Rush Hour (All other hours and weekends)")
    print("--------------------------------------------------")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        filtered_df = df[df["rush_hour"] == 1]
        logger.info(
            f"Filtered dataset by Rush Hour (1): {len(filtered_df)} matches found."
        )
        display_traffic_summary(
            filtered_df["traffic_volume"],
            label="Rush Hour Traffic"
        )
    elif choice == "2":
        filtered_df = df[df["rush_hour"] == 0]
        logger.info(
            f"Filtered dataset by Non-Rush Hour (0): {len(filtered_df)} matches found."
        )
        display_traffic_summary(
            filtered_df["traffic_volume"],
            label="Non-Rush Hour Traffic"
        )
    else:
        print("\n[!] Invalid selection. Returning to main menu.")
        logger.warning(f"Invalid rush hour choice entered: '{choice}'.")

def run_cli_app(df: pd.DataFrame):
    # Main interactive loop handling user inputs and dispatching tasks
    logger.info("Traffic Analytics CLI Application started.")
    
    while True:
        display_menu()
        user_choice = input("Enter your choice (1-4): ").strip()
        
        if user_choice == "1":
            logger.info("User selected Option 1: View Overall Traffic Summary Statistics.")
            display_traffic_summary(df["traffic_volume"], label="All 48,187 Records")
            
        elif user_choice == "2":
            logger.info("User selected Option 2: Filter Traffic by Weather Condition.")
            handle_weather_filter(df)
            
        elif user_choice == "3":
            logger.info("User selected Option 3: Filter Traffic by Rush Hour Status.")
            handle_rush_hour_filter(df)
            
        elif user_choice == "4":
            logger.info("User selected Option 4: Exit Application.")
            print("\nExiting Smart City Traffic Analytics System. Goodbye!")
            break
            
        else:
            print("\n[!] Invalid selection. Please enter a valid number between 1 and 4.")
            logger.warning(f"Invalid menu input received: '{user_choice}'.")


if __name__ == "__main__":
    setup_logging()

    # Accommodate running from repository root or part2_python folder
    csv_path = "part2_python/traffic_features.csv"
    if not os.path.exists(csv_path) and os.path.exists("traffic_features.csv"):
        csv_path = "traffic_features.csv"

    try:
        traffic_data = load_dataset(csv_path)
        run_cli_app(traffic_data)
    except Exception as err:
        logger.error(f"CLI application terminated unexpectedly: {err}", exc_info=True)
        sys.exit(1)