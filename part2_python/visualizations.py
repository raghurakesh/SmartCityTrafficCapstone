# Task 3: Data Visualizations (Matplotlib and Seaborn)

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Module-level logger (isolated from global root logger)
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

    # Handler to write log messages into physical file on disk (mode 'a' appends to pipeline log)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_feature_data(file_path: str) -> pd.DataFrame:
    # Loads the enriched traffic CSV file with error handling
    # Input Parameter : file_path (str) - Path to the CSV file
    # Output : DataFrame loaded with the CSV file
    if not os.path.exists(file_path):
        logger.error(f"Input file not found at path: {file_path}", exc_info=True)
        raise FileNotFoundError(f"Missing input dataset: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(
            f"Dataset loaded for visualization. Shape: "
            f"{df.shape[0]} rows and {df.shape[1]} columns."
        )
        return df
    except Exception as e:
        logger.error(f"Unexpected error loading '{file_path}': {e}", exc_info=True)
        raise

def plot_hourly_traffic_by_day_type(df: pd.DataFrame, output_dir: str):
    # Generates a line plot comparing average hourly traffic on weekdays vs weekends
    # Input parameters:
    #   df (pd.DataFrame): Enriched dataset containing 'hour', 'is_weekend', 'traffic_volume'
    #   output_dir (str): Directory where the PNG file will be saved
    
    # Calculate average traffic volume grouped by hour and day type
    hourly_traffic = (
        df.groupby(["hour", "is_weekend"])["traffic_volume"]
        .mean()
        .reset_index()
    )

    # Separate weekday (is_weekend == 0) and weekend (is_weekend == 1)
    weekday_data = hourly_traffic[hourly_traffic["is_weekend"] == 0]
    weekend_data = hourly_traffic[hourly_traffic["is_weekend"] == 1]

    # Create figure
    plt.figure(figsize=(10, 5))
    
    plt.plot(
        weekday_data["hour"],
        weekday_data["traffic_volume"],
        marker="o",
        label="Weekday (Mon-Fri)",
        color="#1f77b4",
        linewidth=2
    )
    plt.plot(
        weekend_data["hour"],
        weekend_data["traffic_volume"],
        marker="s",
        label="Weekend (Sat-Sun)",
        color="#ff7f0e",
        linewidth=2
    )

    plt.title("Average Hourly Traffic Demand: Weekday vs. Weekend", fontsize=13, pad=12)
    plt.xlabel("Hour of the Day (0–23)", fontsize=11)
    plt.ylabel("Average Traffic Volume (Vehicles / Hour)", fontsize=11)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Day Type")
    plt.tight_layout()

    # Save to disk
    figure_path = os.path.join(output_dir, "01_hourly_traffic_weekday_vs_weekend.png")
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # Mandatory requirement: Log an INFO message when figure is saved with its path
    logger.info(f"Figure successfully saved to '{figure_path}'.")

def plot_traffic_by_weather(df: pd.DataFrame, output_dir: str):
    # Generates a horizontal bar chart showing average traffic volume across weather conditions
    # Input parameters:
    #   df (pd.DataFrame): Enriched dataset containing 'weather_main' and 'traffic_volume'
    #   output_dir (str): Directory where the PNG file will be saved

    # Calculate mean traffic volume per weather category and sort ascending for horizontal display
    weather_summary = (
        df.groupby("weather_main")["traffic_volume"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    plt.figure(figsize=(9, 5))
    bars = plt.barh(
        weather_summary["weather_main"],
        weather_summary["traffic_volume"],
        color="#2ca02c",
        edgecolor="#1b611b"
    )

    # Add numeric labels to the end of each bar for clarity
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 40,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            va="center",
            ha="left",
            fontsize=9
        )

    plt.title("Average Traffic Volume by Weather Condition", fontsize=13, pad=12)
    plt.xlabel("Average Traffic Volume (Vehicles / Hour)", fontsize=11)
    plt.ylabel("Weather Condition", fontsize=11)
    plt.xlim(0, weather_summary["traffic_volume"].max() * 1.15)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save to disk
    figure_path = os.path.join(output_dir, "02_traffic_by_weather_condition.png")
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # Log figure saved milestone
    logger.info(f"Figure successfully saved to '{figure_path}'.")

def plot_congestion_category_distribution(df: pd.DataFrame, output_dir: str):
    # Generates a box plot showing traffic volume distribution across the 4 quartile congestion categories
    # Input parameters:
    #   df (pd.DataFrame): Enriched dataset containing 'congestion_category' and 'traffic_volume'
    #   output_dir (str): Directory where the PNG file will be saved

    category_order = ["Low", "Medium", "High", "Severe"]
    palette_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="congestion_category",
        y="traffic_volume",
        hue="congestion_category",
        order=category_order,
        palette=palette_colors,
        legend=False
    )

    plt.title("Traffic Volume Distribution Across Congestion Categories (Quartile Buckets)", fontsize=13, pad=12)
    plt.xlabel("Congestion Category", fontsize=11)
    plt.ylabel("Traffic Volume (Vehicles / Hour)", fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save to disk
    figure_path = os.path.join(output_dir, "03_traffic_by_congestion_category.png")
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # Log figure saved milestone
    logger.info(f"Figure successfully saved to '{figure_path}'.")

if __name__ == "__main__":
    setup_logging()

    input_data_path = "part2_python/traffic_features.csv"
    figures_dir = "part2_python/figures"
    os.makedirs(figures_dir, exist_ok=True)

    try:
        traffic_df = load_feature_data(input_data_path)

        # Plot 1: Weekday vs Weekend Hourly Pattern
        plot_hourly_traffic_by_day_type(traffic_df, figures_dir)  
    
        # Plot 2: Weather Condition Impact on Traffic
        plot_traffic_by_weather(traffic_df, figures_dir)  
        
        # Plot 3: Traffic Volume Distribution by Congestion Category
        plot_congestion_category_distribution(traffic_df, figures_dir)    
    except Exception as err:
        logger.error(f"Failed to start visualization pipeline: {err}", exc_info=True)
        sys.exit(1)