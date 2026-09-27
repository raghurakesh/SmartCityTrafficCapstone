-- Total number of rows in the database

SELECT COUNT(*) as total_no_of_rows
FROM "Metro_Interstate_Traffic_Volume";

-- Total number of houws, total traffic volume, average traffic volume per year.

SELECT
substr(date_time, 1, 4) AS year,
count(*) AS total_hours_recorded,
sum(traffic_volume) AS total_traffic_volume,
round(avg(traffic_volume), 2) as average_hourly_volume
FROM "Metro_Interstate_Traffic_Volume"
WHERE substr(date_time, 1, 4) >= '2012'
AND substr(date_time, 1, 4) <= '2017'
GROUP BY substr(date_time, 1, 4)
ORDER BY year;

-- Temperature patterns for 2015, 2016 and 2017 during: New Year's Day, Labor Day

SELECT
holiday,
substr(date_time, 1, 4) AS year,
date_time,
temp as temperature,
round (temp - 273.15,2) as temperature_celsius,
traffic_volume
FROM "Metro_Interstate_Traffic_Volume"
WHERE holiday IN ('New Years Day', 'Labor Day')
AND substr(date_time, 1, 4) BETWEEN '2015' AND '2017'
ORDER BY holiday, year;
