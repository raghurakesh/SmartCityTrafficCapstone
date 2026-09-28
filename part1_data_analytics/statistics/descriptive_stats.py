# Import pandas and numpy to perform statistical analysis
import pandas as pd
import numpy as np

# State the location of the csv file on which the analysis has to be done and read it
csv_location = 'Metro_Interstate_Traffic_Volume.csv'
df = pd.read_csv(csv_location)

# Calculate the Mean, Median, Standard Deviation, Variance, Min, Max and Range of the traffic Volme

mean_volume = df['traffic_volume'].mean()
median_volume = df['traffic_volume'].median()
stddev  = df['traffic_volume'].std()
variance = df['traffic_volume'].var()
min_volume = df['traffic_volume'].min()
max_volume  = df['traffic_volume'].max()
range_of_volume = max_volume - min_volume

# Print the results

print(f"Mean:               {mean_volume:.2f} veh/hr")
print(f"Median:             {median_volume:.2f} veh/hr")
print(f"Standard Deviation: {stddev:.2f} veh/hr")
print(f"Variance:           {variance:.2f}")
print(f"Range:              {range_of_volume} veh/hr, (Min: {min_volume} veh/hr, Max: {max_volume} veh/hr)")


# Calculate Correlation Coefficient

corr_coef = df['temp'].corr(df['traffic_volume'])

print(f"Correlation Coefficient (temp vs traffic_volume): {corr_coef:.4f}")