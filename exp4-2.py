import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import math

# Load Data (Replace with local path if not using Colab)
df = pd.read_excel('regdata.xlsx')
df2 = df[['Price', 'Dem']].copy()
df2['naturalLogPrice'] = np.log(df2['Price'])
df2['naturalLogDemand'] = np.log(df2['Dem'])

# Visualization
sns.regplot(x="naturalLogPrice", y="naturalLogDemand", data=df2, fit_reg=True)
plt.title('Log-Log Regression')
plt.show()

# Prepare Data
X = df2[['naturalLogPrice']]
y = df2['naturalLogDemand']

# Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print("Predicted values:", y_pred)

# Evaluation Metrics
corr, _ = pearsonr(df2['naturalLogPrice'], df2['naturalLogDemand'])
print(f"Pearson's correlation: {corr:.3f}")

n = len(y)
mse = np.mean((y - y_pred) ** 2)
rmse = math.sqrt(mse)
R2 = model.score(X, y)
rmsre = math.sqrt(np.mean(((y - y_pred) / y) ** 2))
mae = np.mean(np.abs(y - y_pred))
mape = np.mean(np.abs((y - y_pred) / y)) * 100

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Coefficient of Determination (R²): {R2}")
print(f"Root Mean Squared Relative Error: {rmsre}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Percentage Error: {mape}")
