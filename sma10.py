import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
from sklearn.cluster import KMeans
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\anish\Downloads\sma_ques10_dataset.csv")

G = nx.Graph()
G.add_edges_from(zip(df["source"], df["target"]))

# --- Girvan-Newman Community Detection ---
communities = girvan_newman(G)
top_level_communities = next(communities)
community_list = [list(c) for c in top_level_communities]
print("Detected Communities:", community_list)

# Draw the graph with communities
color_map = {}
colors = ["red", "blue", "green", "purple", "orange"]
for i, community in enumerate(community_list):
    for node in community:
        color_map[node] = colors[i % len(colors)]

node_colors = [color_map.get(node, "gray") for node in G.nodes()]
nx.draw(G, with_labels=True, node_color=node_colors, node_size=500)
plt.title("Community Detection using Girvan-Newman")
plt.show()

# --- KMeans Clustering on Centrality Features ---
# Compute centralities
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Create a feature matrix
features = []
nodes = list(G.nodes())
for node in nodes:
    features.append([
        degree_centrality[node],
        betweenness_centrality[node]
    ])

features = np.array(features)

# Run KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(features)

# Plot KMeans clustering
plt.figure(figsize=(8, 5))
plt.scatter(features[:, 0], features[:, 1], c=labels, s=100)
for i, node in enumerate(nodes):
    plt.text(features[i, 0], features[i, 1], node, fontsize=9, ha='right')
plt.xlabel("Degree Centrality")
plt.ylabel("Betweenness Centrality")
plt.title("KMeans Clustering of Nodes Based on Centrality")
plt.show()


# Sort nodes by betweenness centrality
sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
print("\nTop Influential Nodes (by Betweenness Centrality):")
for node, score in sorted_nodes[:5]:
    print(f"{node}: {score:.4f}")
