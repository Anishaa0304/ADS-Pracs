# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import datasets

# Load dataset
data = pd.read_csv(r"C:\Users\anish\Downloads\ADS lab exam solutions\ADS Datasets\Iris.csv")

# Drop the 'Id' column and visualize boxplots
new_data = data.drop(['Id'], axis=1)
new_data.boxplot()
plt.title("Boxplot of Iris Dataset Features")
plt.show()

# Prepare data for k-NN outlier detection
X = new_data.drop('Species', axis=1).values
nbrs = NearestNeighbors(n_neighbors=3)
nbrs.fit(X)
distances, indexes = nbrs.kneighbors(X)

# Plot mean distances
plt.plot(distances.mean(axis=1))
plt.title("Mean k-NN Distances")
plt.xlabel("Data Points")
plt.ylabel("Mean Distance")
plt.show()

# Get outlier indices based on a threshold
outlier_index = np.where(distances.mean(axis=1) > 0.3)
outlier_values = new_data.iloc[outlier_index]
print("Outliers detected using k-NN:\n", outlier_values)

# DBSCAN clustering using SepalLengthCm and SepalWidthCm
df = data[["SepalLengthCm", "SepalWidthCm"]]
model = DBSCAN(eps=0.4, min_samples=10).fit(df)

# Visualize clustering results
colors = model.labels_
plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"], c=colors, cmap='rainbow')
plt.title("DBSCAN Clustering on Iris Sepal Data")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

# Extract and print DBSCAN outliers
outliers = data[model.labels_ == -1]
print("Outliers detected using DBSCAN:\n", outliers)
