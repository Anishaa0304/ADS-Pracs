import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")

df['polarity'] = df['Content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

df['engagement'] = (df['Likes'] + df['Comments'] + df['Shares']) / df['Followers']

#According to Content_Type
plt.figure(figsize=(8,5))
sns.barplot(data = df, x = 'Content_Type', y = 'engagement', palette='viridis')
plt.title("Average Engagement according to Content Type")
plt.ylabel("Engagement Rate")
plt.xlabel("Content Type")
plt.show()

#According to Sentiment
plt.figure(figsize=(8,5))
sns.boxplot(data = df, x="sentiment", y = 'engagement', palette = 'coolwarm')
plt.title("Engagement Rate by Sentiment")
plt.ylabel("Engagement Rate")
plt.xlabel("Sentiment")
plt.show()
