import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")

# Convert 'Post_Date' to datetime format
df['Post_Date'] = pd.to_datetime(df['Post_Date'])

# Extract year, month, and day for trend analysis
df['Year'] = df['Post_Date'].dt.year
df['Month'] = df['Post_Date'].dt.month
df['Day'] = df['Post_Date'].dt.day

# 1. Trend of Total Likes over Time (Monthly)
monthly_engagement = df.groupby(['Year', 'Month'])['Likes'].sum()

# Plotting Total Likes over Time (Monthly)
plt.figure(figsize=(12, 6))
monthly_engagement.plot(kind='line', marker='o', color='green')
plt.title('Total Likes Trend Over Time (Monthly)')
plt.xlabel('Year-Month')
plt.ylabel('Total Likes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# . Trend of Average Comments over Time (Monthly)
monthly_comments = df.groupby(['Year', 'Month'])['Comments'].mean()

# Plotting Average Comments over Time (Monthly)
plt.figure(figsize=(12, 6))
monthly_comments.plot(kind='line', marker='o', color='orange')
plt.title('Average Comments Trend Over Time (Monthly)')
plt.xlabel('Year-Month')
plt.ylabel('Average Comments')
plt.xticks(rotation=45)
plt.show()

# 3. Trend of Total Likes over Hashtags
# Ensure Hashtags are treated properly (handle missing or NaN values)
df['Hashtags'] = df['Hashtags'].fillna('')  # Handle NaN values

# Split the hashtags by spaces (or other delimiters), convert to lowercase, and strip extra spaces
hashtags = df['Hashtags'].str.lower().str.split('#').sum()  # Split by '#' and flatten the list

# Clean up the hashtags by removing empty strings
hashtags = [hashtag.strip() for hashtag in hashtags if hashtag.strip() != '']

# Count the frequency of each hashtag
hashtag_counts = pd.Series(hashtags).value_counts()

# Display the top 10 hashtags
print("Top 10 Hashtags:", hashtag_counts.head(10))

# Plotting the Total Likes by Hashtag (Top 10 Hashtags)
plt.figure(figsize=(12, 6))
hashtag_counts.head(10).plot(kind='bar', color='green')
plt.title('Top 10 Hashtags by Frequency')
plt.xlabel('Hashtags')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
