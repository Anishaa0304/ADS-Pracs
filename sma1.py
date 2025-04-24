import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")
# Create a TfidfVectorizer to extract keywords
vectorizer = TfidfVectorizer(max_features=10, stop_words='english')

# Fit and transform the 'Content' column
X = vectorizer.fit_transform(df['Content']) #not useful

# Get the top 10 keywords
keywords = vectorizer.get_feature_names_out()

# Display the top keywords
print("Top Keywords:", keywords)
