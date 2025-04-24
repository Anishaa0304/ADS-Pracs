from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")

# Create a CountVectorizer to extract keywords
vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
X = vectorizer.fit_transform(df['Content'])

# Get the top 10 most frequent keywords
keywords = vectorizer.get_feature_names_out()

print("Top 10 Keywords:", keywords)

# Visualizing the most frequent keywords
keyword_counts = X.sum(axis=0).A1  # Sum of word frequencies across all documents
plt.figure(figsize=(10, 6))
plt.bar(keywords, keyword_counts, color='lightgreen')
plt.title('Top 10 Most Mentioned Keywords')
plt.xlabel('Keywords')
plt.ylabel('Frequency')
plt.show()
