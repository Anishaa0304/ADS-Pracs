import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

df = pd.read_csv(r"C:\Users\anish\Downloads\ADS lab exam solutions\ADS Datasets\supermarket_sales - Sheet1.csv")

print(df.head())
print(df.info())
print(df.describe())

# Median for numeric columns only
median_values = df.select_dtypes(include='number').median() #if i dont include number it will give error
print("Median values:\n", median_values)

# Modes
print("Mode of Product line:", df['Product line'].mode()[0])
print("Mode of City:", df['City'].mode()[0])
print("Mode of Payment:", df['Payment'].mode()[0])
print("Mode of Customer type:", df['Customer type'].mode()[0])
print("Mode of Gender:", df['Gender'].mode()[0])

# Scatter plots
plt.scatter(df['Tax 5%'], df['Unit price'], c="blue")
plt.title("Tax vs Unit Price")
plt.xlabel("Tax 5%")
plt.ylabel("Unit price")
plt.show()

plt.scatter(df['gross income'], df['Unit price'], c="blue")
plt.title("Gross Income vs Unit Price")
plt.xlabel("Gross Income")
plt.ylabel("Unit price")
plt.show()

plt.scatter(df['Quantity'], df['Total'], c="blue")
plt.title("Quantity vs Total")
plt.xlabel("Quantity")
plt.ylabel("Total")
plt.show()

# Box plots
data = df[['Tax 5%', 'gross income', 'Rating']]
data.plot(kind='box', title='Boxplot of Selected Columns')
plt.show()

plt.boxplot(df['Total'])  # multiple columns in box plot : plt.boxplot([],[])
plt.title("Boxplot of Total")  # plt.title(), plt.xlabel(), plt.ylabel(), plt.xticks(), plt.yticks()
plt.show()

# Trimmed mean
trimmed_mean = stats.trim_mean(df['Total'], 0.1)
print("Trimmed mean of Total:", trimmed_mean)

# Sum of 'Total' column
total_sum = df['Total'].sum()
print("Sum of Total:", total_sum)

# Frequency of Product line
count = df['Product line'].value_counts()
print("Product line frequency:\n", count)

# Variance
print("Variance:\n", df.select_dtypes(include='number').var())

# Correlation matrix
print("Correlation matrix:\n", df.select_dtypes(include='number').corr())

# Standard error of mean
print("Standard error of mean:\n", df.select_dtypes(include='number').sem())

# Sum of squares
sos = sum(val * val for val in df['Total'])
print("Sum of squares (Total):", sos)

# Skewness
print("Skewness:\n", df.select_dtypes(include='number').skew())

# Kurtosis
sr = pd.Series(df['Total'])
print("Kurtosis of Total:", sr.kurtosis())

# Histogram with KDE
sns.histplot(df['Total'], kde=True)
plt.title("Distribution of Total")
plt.xlabel("Total")
plt.ylabel("Frequency")
plt.show()
