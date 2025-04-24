import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")
print(df.head())
print(df.info())

# Bar Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Content_Type', y='Likes', estimator='sum', hue='Content_Type')
plt.title('Total Likes by Content Type')
plt.xlabel('Content Type')
plt.ylabel('Total Likes')
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Comments', y='Likes', hue='Content_Type', s=100)
plt.title('Likes vs Comments')
plt.xlabel('Comments')
plt.ylabel('Likes')
plt.legend(title='Content Type')
plt.show()

# Bubble Chart: Likes vs Shares (Size = Followers)
plt.figure(figsize=(8, 5))
plt.scatter(df['Shares'], df['Likes'], s=df['Followers']/10, alpha=0.6, c='teal', edgecolors='w')
plt.title('Likes vs Shares (Bubble size = Followers)')
plt.xlabel('Shares')
plt.ylabel('Likes')
plt.show()

# Heatmap of Correlation
plt.figure(figsize=(6, 4))
sns.heatmap(df[['Likes', 'Comments', 'Shares', 'Followers']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Numeric Features')
plt.show()

# Network Graph: Hashtag of Co-occurrence
# Extract hashtags from each post
df['Hashtags'] = df['Hashtags'].astype(str)
df['Hashtag_List'] = df['Hashtags'].apply(lambda x: [h.strip() for h in x.split()])

# Create edges
edges = []
for tags in df['Hashtag_List']:
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            edges.append((tags[i], tags[j]))

# Build graph
G = nx.Graph()
G.add_edges_from(edges)
plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=1000, font_size=10)
plt.title('Hashtag Co-occurrence Network')
plt.show()

# WordCloud
# Combine all hashtags into one string
hashtags = ' '.join(df['Hashtags'].astype(str).tolist())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(hashtags)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Hashtags')
plt.show()
