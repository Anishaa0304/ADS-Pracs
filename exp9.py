import pandas as pd
import numpy as np
import math
import statistics


# Load the dataset
df = pd.read_csv(r"C:\Users\anish\Downloads\ADS lab exam solutions\ADS Datasets\supermarket_sales - Sheet1.csv")

# -------------------------------------
# One-Sample Z-test and T-test on "Total"
# -------------------------------------

# Sample for T-test (n < 30) and Z-test (n > 30)
df_t = df.sample(n=28)
df_z = df.sample(n=100)

# Population values
mu = df["Total"].mean()
sigma = df["Total"].std()

# Sample values
x_bar_t = df_t["Total"].mean()
s_t = df_t["Total"].std()
nt = len(df_t)

x_bar_z = df_z["Total"].mean()
nz = len(df_z)

# Z-test
print("\n--- One-Sample Z-Test ---")
z_score = (x_bar_z - mu) / (sigma / math.sqrt(nz))
print("Z-Score:", z_score)
z_critical = 1.65
if z_score > z_critical:
    print("Result: Reject Null Hypothesis")
else:
    print("Result: Do NOT Reject Null Hypothesis")

# T-test
print("\n--- One-Sample T-Test ---")
t_score = (x_bar_t - mu) / (s_t / math.sqrt(nt))
print("T-Score:", t_score)
t_critical = 1.703
if t_score > t_critical:
    print("Result: Reject Null Hypothesis")
else:
    print("Result: Do NOT Reject Null Hypothesis")

# -------------------------------------
# Two-Sample T-Test on "Quantity" by Gender
# -------------------------------------

df_gender_ttest = df.sample(n=29)
men_t = df_gender_ttest[df_gender_ttest["Gender"] == "Male"]["Quantity"]
women_t = df_gender_ttest[df_gender_ttest["Gender"] == "Female"]["Quantity"]

men_mean = statistics.fmean(men_t)
women_mean = statistics.fmean(women_t)

men_std = statistics.stdev(men_t, xbar=men_mean)
women_std = statistics.stdev(women_t, xbar=women_mean)

n_men = len(men_t)
n_women = len(women_t)

pooled_std = math.sqrt((((n_men - 1) * men_std**2) + ((n_women - 1) * women_std**2)) / (n_men + n_women - 2))
t_score_2samp = abs(men_mean - women_mean) / (pooled_std * math.sqrt(1/n_men + 1/n_women))

print("\n--- Two-Sample T-Test (Gender, Quantity) ---")
print("T-Score:", t_score_2samp)
if t_score_2samp > t_critical:
    print("Result: Reject Null Hypothesis")
else:
    print("Result: Do NOT Reject Null Hypothesis")

# -------------------------------------
# Two-Sample Z-Test on "Quantity" by Gender (n = 100)
# -------------------------------------

df_gender_ztest = df.sample(n=100)
men_z = df_gender_ztest[df_gender_ztest["Gender"] == "Male"]["Quantity"]
women_z = df_gender_ztest[df_gender_ztest["Gender"] == "Female"]["Quantity"]

men_mean_z = statistics.fmean(men_z)
women_mean_z = statistics.fmean(women_z)

men_std_z = statistics.stdev(men_z, xbar=men_mean_z)
women_std_z = statistics.stdev(women_z, xbar=women_mean_z)

n_men_z = len(men_z)
n_women_z = len(women_z)

z_score_2samp = abs(men_mean_z - women_mean_z) / math.sqrt((men_std_z**2 / n_men_z) + (women_std_z**2 / n_women_z))
z_critical_2samp = 1.645

print("\n--- Two-Sample Z-Test (Gender, Quantity) ---")
print("Z-Score:", z_score_2samp)
if z_score_2samp > z_critical_2samp:
    print("Result: Reject Null Hypothesis")
else:
    print("Result: Do NOT Reject Null Hypothesis")
