# Part 1 - Task 3: Probability and Congestion Analysis

# Definition of congestion : traffic volume > 5,500 vehicles.

---

## Task 3.1: BASIC PROBABILITY

Total Rows: 48204
P(Congestion): 0.1473 (14.73%)
P(Clear Weather): 0.2778 (27.78%)
P(Congestion AND Clear): 0.0366 (3.66%)

# Intepretation

1. Approx 15% probability of severe congeston, i.e., Around 15 (14.73) hours in every 100 hours.
2. Approx 28% probability of clear weather, i.e., Around 28 (27.78) hours in every 100 hours.
3. Both conditions happening together, i.e., a clear weather with severe congestion is only about 3.7% (3.66%)

---

## TASK 3.2: CONDITIONAL PROBABILITY & INDEPENDENCE

P(Clear Weather | Congestion): 0.2483 (24.83%)
P(High Temp | Congestion): 0.2630 (26.30%)
P(Congestion) \* P(Clear): 0.0409 (4.09%)
Odds of Congestion (Clear): 0.1516 (15.16%)
Odds of Congestion (Cloudy): 0.2062 (20.62%)
Odds Ratio (Clear vs Cloudy): 0.7354 (73.54%)

# Intepretation

1. Is the weather conditions and congestion independent
   - If two events are independent, P(Congestion AND Clear) must equal P(Congestion) \* P(Clear). Here, P(Congestion AND Clear) = 0.0366, whereas P(Congestion) \* P(Clear) = 0.0409. Since $0.0366 is not equal to 0.0409$, weather condition and traffic congestion are not independent.

2. Odds ratio of congestion in clear versus cloudy weather
   - The odds ratio is 0.7354. As the value is less than 1, the odds of congestion during clear weather is around 26.5% lower compared to cloudy weather (1 - 0.7354 = 0.26.46)

3. High Temperature Under Congestion
   - When the congestion is sever, higher warm temperatures are present around 26.30% times. This also matches with the weak correlation observed earlier.
