from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Load Iris dataset
df = datasets.load_iris()
X = df.data
y = df.target

# Convert to DataFrame
dfr = pd.DataFrame(data=X, columns=df.feature_names)
dfr['target'] = y

# Apply KMeans Clustering (k=3)
kmeans_model = KMeans(n_clusters=3, n_init=10, random_state=42)
dfr['kmeans_3'] = kmeans_model.fit_predict(dfr[['sepal length (cm)', 'target']])

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(x=dfr['sepal length (cm)'], y=dfr['target'], c=dfr['kmeans_3'], cmap='viridis')
plt.title('KMeans Clustering (k=3) on Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Target')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Evaluation Metrics
sil_score = silhouette_score(dfr[['sepal length (cm)', 'target']], dfr['kmeans_3'], metric='euclidean')
ari_score = adjusted_rand_score(dfr['target'], dfr['kmeans_3'])
nmi_score = normalized_mutual_info_score(dfr['target'], dfr['kmeans_3'])

print(f"Silhouette Score: {sil_score}")
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information Score: {nmi_score}")
