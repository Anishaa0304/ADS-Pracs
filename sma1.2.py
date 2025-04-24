from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")

# Create a TfidfVectorizer and transform the content
vectorizer = TfidfVectorizer(stop_words='english') #isme features limit nhi kiya
X = vectorizer.fit_transform(df['Content'])

# Perform LDA (Topic Modeling)
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

# Display the topics
for idx, topic in enumerate(lda.components_):
    print(f"Topic #{idx}:")
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]  # sorts the indices of topics in ascending order of importance
    print(", ".join(top_words))
