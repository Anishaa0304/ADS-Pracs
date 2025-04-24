import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r"C:\Users\anish\Downloads\sma_ques9_dataset.csv")

# Basic summary
print("Summary by Competitor:")
print(df.groupby("competitor")[["likes", "comments", "shares"]].mean())

# Count of posts per platform per competitor
platform_dist = df.groupby(["competitor", "platform"]).size().unstack()
print("\nPosts by Platform:")
print(platform_dist)

# Sentiment distribution
sentiment_dist = df.groupby(["competitor", "sentiment"]).size().unstack(fill_value=0)
print("\nSentiment Distribution:")
print(sentiment_dist)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="competitor", y="likes", estimator='mean', ci=None)
plt.title("Average Likes by Competitor")
plt.ylabel("Average Likes")
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="competitor", hue="sentiment")
plt.title("Sentiment Distribution by Competitor")
plt.ylabel("Number of Posts")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="competitor", y="shares")
plt.title("Shares Distribution by Competitor")
plt.ylabel("Shares")
plt.show()
