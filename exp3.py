import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Read dataset
df = pd.read_csv(r"C:\Users\anish\Downloads\ADS lab exam solutions\ADS Datasets\supermarket_sales - Sheet1.csv")

# Display first few rows of the dataframe
df.head()

# Scatter plot
plt.scatter(df['Tax 5%'], df['Unit price'], c="blue")
plt.show()

# BoxPlot
x2 = df['Tax 5%']
x4 = df['gross income']
x5 = df['Rating']
data = pd.DataFrame({"Tax 5%": x2, "gross income": x4, "Rating": x5})
ax = data[['Tax 5%', 'gross income', 'Rating']].plot(kind='box', title='boxplot')
plt.show()

# Distribution Chart / Distplot (Warning due to deprecation)
g = sns.distplot(df['Total'])
plt.show()

# JointPlot
sns.jointplot(x='Total', y='Tax 5%', data=df)
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Histogram
df['Rating'].hist()
plt.show()

# Pie chart for 'Product line'
lst = df['Product line'].unique()
t = [23, 17, 35, 29, 12, 41]
plt.pie(t, labels=lst, autopct='%1.1f%%', shadow=True)
plt.show()

# Density Chart
df['Rating'].plot.density(color='green')
plt.title('Density plot for Rating')
plt.show()

# Scatter Matrix
pd.plotting.scatter_matrix(df)
plt.show()

# Rugplot
plt.figure(figsize=(15, 5))
sns.rugplot(data=df, x="Total")
plt.show()

# Column chart (Bar plot between 2 attributes)
df1 = df.head(10)
df1.plot.bar()
plt.bar(df1['Gender'], df1['Total'])
plt.xlabel("Gender")
plt.ylabel("Total")
plt.show()

# Line plot with Plotly
df1 = df.head(15)
fig = px.line(df1, x="Date", y="Total", color='City')
fig.show()

# Bubble Chart with Plotly
fig = px.scatter(df1, x="Total", y="Tax 5%", size="Quantity", color="City", hover_name="Product line", log_x=True, size_max=60)
fig.show()

# Parallel Coordinates with Plotly
df1 = df.sample(n=100)
fig = px.parallel_coordinates(df1, color="Total",
                               dimensions=['Quantity', 'Unit price', 'Rating'],
                               color_continuous_scale=px.colors.diverging.Tealrose,
                               color_continuous_midpoint=2)
fig.show()

# Creating Andrews curves
df1 = df[['Quantity', 'Total', 'Rating']]
df1 = df1.sample(n=50)
x = pd.plotting.andrews_curves(df1, 'Rating')
x.plot()
plt.show()

# Heatmap with Plotly
fig = px.imshow(df1)
fig.show()

# Another Line plot with Plotly
fig = px.line(df1, x='Quantity', y="Total")
fig.show()
