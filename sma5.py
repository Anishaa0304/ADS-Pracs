import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")


def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity


df['polarity'] = df['Content'].apply(get_sentiment)


def classify_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'


df['sentiment'] = df['polarity'].apply(classify_sentiment)

plt.figure(figsize=(10,6))
sns.countplot(data=df, x="sentiment", hue=None, palette="coolwarm", legend=False)
plt.title('Sentiment Classification')
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.show()

plus = ' '.join(df[df['sentiment'] == 'Positive']['Content'].dropna().astype(str))
minus = ' '.join(df[df['sentiment'] == 'Negative']['Content'].dropna().astype(str))

wordcloud_pos = WordCloud(
    width = 800, height = 400, background_color = 'white', colormap="Greens"
    ).generate(plus)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Content WordCloud")
plt.show()

wordcloud_neg = WordCloud(
    width=800, height=400, background_color="black", colormap="Reds"
).generate(minus)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Content WordCloud")
plt.show()
