import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")

print(df.columns)
print(df.head())
print(df.info())

hash = 'Hashtags'
# Convert hashtags string to list
df["hashtags"] = df["Hashtags"].fillna("").apply(lambda x: x.lower().split())

# Flatten the list of all hashtags
all_hashtags = [tag for sublist in df["hashtags"] for tag in sublist]

# Count frequency of hashtags
hashtag_counts = Counter(all_hashtags)
top_hash = hashtag_counts.most_common(4)

hashtag_df = pd.DataFrame(top_hash, columns=['Hashtag', 'Count'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(hashtag_counts)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Top Hashtags WordCloud")
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=hashtag_df, x = 'Count', y = 'Hashtag')
plt.title("Top 4 hashtags")
plt.xlabel("Frequency")
plt.ylabel("Hashtag")
plt.show()
