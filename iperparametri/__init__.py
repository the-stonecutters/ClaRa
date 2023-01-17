import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

def load_XY():
    stopwords = nltk.corpus.stopwords.words('italian')
    stopwords.extend(['ansa'])

    classificate = pd.read_csv('news.csv')
    classificate = classificate[classificate['news_body'].notnull()]

    vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3))

    X = vectorizer.fit_transform(classificate['news_body'])
    Y = classificate['news_category']

    return X, Y