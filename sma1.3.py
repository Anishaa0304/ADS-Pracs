from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")

# Create a WordCloud from the content of the posts
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Content']))

# Display the WordCloud
plt.figure(figsize=(10, 6))  #fig size in inches
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # removes x y axis
plt.show()
