#  Task 3: Probability and Congestion Analysis
# Import pandas library
import pandas as pd

# Load the dataset and print total number of records
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
total_no_of_records = len(df)

# Define the conditions to be checked.
is_congestion = df['traffic_volume'] > 5500
is_clear = df['weather_main'] == 'Clear'
is_cloudy = df['weather_main'] == 'Clouds'
is_high_temp = df['temp'] > 292.0

# Calculate P(Congestion), P(Clear Weather), P(Congestion AND Clear Weather) 
p_congestion = is_congestion.mean()
p_clear = is_clear.mean()
p_congestion_and_clear = (is_congestion & is_clear).mean()

# P(Clear | Congestion) = Count(Clear AND Congestion) / Count(Congestion)
p_clear_given_cong = (is_congestion & is_clear).sum() / is_congestion.sum()

# P(High Temp | Congestion) = Count(High Temp AND Congestion) / Count(Congestion)
p_high_temp_given_cong = (is_congestion & is_high_temp).sum() / is_congestion.sum()

# Check if weather conditions and congestion can be considered independent 
p_product = p_congestion * p_clear

# Odds ratio of congestion: Clear weather vs Cloudy weather
odds_clear = (is_clear & is_congestion).sum() / (is_clear & ~is_congestion).sum()
odds_cloudy = (is_cloudy & is_congestion).sum() / (is_cloudy & ~is_congestion).sum()
odds_ratio = odds_clear / odds_cloudy

# Output
# ---------------------------------------------------------------------
print("---------------------------")
print("Task 3.1: BASIC PROBABILITY")
print("---------------------------")
print(f"Total Rows:                   {total_no_of_records}")
print(f"P(Congestion):                {p_congestion:.4f} ({p_congestion*100:.2f}%)")
print(f"P(Clear Weather):             {p_clear:.4f} ({p_clear*100:.2f}%)")
print(f"P(Congestion AND Clear):      {p_congestion_and_clear:.4f} ({p_congestion_and_clear*100:.2f}%)")
print("------------------------------------------------")
print("TASK 3.2: CONDITIONAL PROBABILITY & INDEPENDENCE")
print("------------------------------------------------")
print(f"P(Clear Weather | Congestion): {p_clear_given_cong:.4f} ({p_clear_given_cong*100:.2f}%)")
print(f"P(High Temp | Congestion):     {p_high_temp_given_cong:.4f} ({p_high_temp_given_cong*100:.2f}%)")
print(f"P(Congestion) * P(Clear):      {p_product:.4f} ({p_product*100:.2f}%)")
print(f"Odds of Congestion (Clear):    {odds_clear:.4f} ({odds_clear*100:.2f}%)")
print(f"Odds of Congestion (Cloudy):   {odds_cloudy:.4f} ({odds_cloudy*100:.2f}%)")
print(f"Odds Ratio (Clear vs Cloudy):  {odds_ratio:.4f} ({odds_ratio*100:.2f}%)")