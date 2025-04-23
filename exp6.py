# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv(r"C:\Users\anish\Downloads\ADS lab exam solutions\ADS Datasets\supermarket_sales - Sheet1.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Filter for Yangon City
data = df[df['City'] == 'Yangon'].copy()

# Drop unnecessary columns
r_col = ['Invoice ID', 'Branch', 'City', 'Customer type', 'Gender',
         'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Time',
         'Payment', 'cogs', 'gross margin percentage', 'gross income', 'Rating']
data.drop(r_col, axis=1, inplace=True)

# Set Date as index
data = data[['Date', 'Total']].sort_values('Date')
data.set_index('Date', inplace=True)
df1 = data.copy()

# Plot Sales
data.plot(figsize=(15, 6), legend=True)
plt.ylabel("Sales", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.title("Date Vs Sales (Yangon)", fontsize=20)
plt.grid(True)
plt.show()

# ADF Test for Stationarity
result = adfuller(data['Total'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Resample to Daily Mean
data = data['Total'].resample('D').mean()

# Seasonal Decomposition
decompose_result_mult = seasonal_decompose(data, model="multiplicative")
decompose_result_mult.plot()
plt.show()

# ACF and PACF
plot_acf(data.dropna(), lags=40)
plt.title("Autocorrelation (ACF)")
plt.show()

plot_pacf(data.dropna(), lags=40, method='ywm')
plt.title("Partial Autocorrelation (PACF)")
plt.show()

# Train-Test Split
inputs = df1.index
target = df1['Total'].copy()
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=1/3, random_state=0)

# Fit ARIMA Model
model = ARIMA(y_train, order=(0, 3, 1))
model_fit = model.fit()

# Forecast
predictions = model_fit.forecast(len(y_test))

# Evaluation
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = sqrt(mse)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
