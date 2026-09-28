# Task 4: Power BI Traffic Intelligence Dashboard

# 4.1 Data quality and preparation

1. Determine the number of rows and columns : 48204 rows and 9 columns (original from file) exists in the file
2. Identify null/missing values and the columns affected : There are 0 null or empty cells existing across any column in the dataset.
3. Check whether the data types are appropriate : All data types set as required.
4. Extract Hour from DateTime : Extracted the Hour from the DateTime and created a new field
5. Create Temperature in Celsius : New field created calculating the temperature in Celcius from Kelvin.
6. Create a Traffic Category : Segmented traffic into three categories as expected

# 4.2 Dashboard analysis

A. Daily traffic trends - created in the Power BI Dashboard
B. Hourly traffic patterns - created column chart as required
C. Weather impact

- Which weather condition has the highest average traffic? - Clouds ( approx 3618 vehicles/hour) followed by Haze ( approx 3,550 vehicles/hour)
- Which has the lowest? - Squall ( approx 2,061 vehicles/hour)
- What is the difference between the highest and lowest average traffic? - Difference between Clouds (3,618 veh/hr) to Squall (2,061 veh/hr) is 1,557 vehicles/hour
  D. Temperature and traffic
- Whether a visible relationship exists : No visible relationship can be seen, the scatter plot shows a broad, vertical cluster spanning across all traffic volumes in normal operating termperatures. This confirms with the correlation coefficient score which was calculated earlier (as low)
- The temperature range associated with higher traffic - Between -5 celcius and +25 celcius
- Any notable outliers - A cluster point can be seen isolated at -273.15 celcius on the far left side of the cluster chart. This looks like a recording error rather than an actual condition.

# 4.3 KPI cards and filters

All required KPI cards and slicers created.

# Data Analytics Insights Report

1. Focus on rush hour or peak time traffic instead of temperature or weather - Commuters travel based on work and school hours irrespective of the temperature, or how hot, warm or cold it is. The city should refine the traffic control measures during this time, for example, adjusting the traffic light timings, during those hours, instead of adjusting them on temperature changes.

2. Severe weather conditions have indicated upto 43% drop in the traffic volume, possibly due to slow driving, visibility or accidents. When such a weather forecast is predicted, the traffic has to be managed, and support and emergency teams have to be available on stand by to help.

3. There are some outlier conditions, and some missing data for certain days and hours, which indicates that possibly there are sensor and monitoring instrument failures. To ensure that the data quality and analysis is correct, these have to be fixed to ensure all the data are correct, and the analysis and action items are accurate.
