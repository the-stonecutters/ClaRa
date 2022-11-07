from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from evaluate import valutazione
from newspaper import Article

import matplotlib.pyplot as plt
import pandas as pd
import nltk

nltk.download('stopwords')
stemmer = nltk.stem.SnowballStemmer('italian')
stopwords = nltk.corpus.stopwords.words('italian')
stopwords.extend(['ansa'])


# Carico i dataset
print('Loading...')
classificate = pd.read_csv('news.csv').drop(columns=['created_at', 'updated_at'])
classificate = classificate[classificate['news_category'] != 'Topnews']
classificate = classificate[classificate['news_body'].notnull()]


vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3))
X = vectorizer.fit_transform(classificate['news_body'])

classifier = SGDClassifier(alpha=1e-5, tol=0.1)
classifier.fit(X, classificate['news_category'])


topnews = pd.read_json('ansa.min.json')
topnews = topnews[topnews['text'].notnull()]

print('Ready')


def classify():
    link = input('Inserisci il link della notizia da classificare: ')
    article = Article(link, language='it')
    article.download()
    article.parse()
    vector = vectorizer.transform([article.text])
    category = classifier.predict(vector)[0]
    print(category)


def classify_ansa():
    X = vectorizer.transform(topnews['text'])
    topnews['category'] = classifier.predict(X)

    topnews['category'].value_counts().plot(kind='bar', ylabel='% (previsione)')
    plt.show()

    while True:
        try:
            c = int(input("Inserisci l'id della notizia da visualizzare, 0 per uscire per uscire: "))
        except ValueError:
            c = -1

        if c == 0:
            return
        elif c < 0 or c > 34490:
            print("Inserisci un numero tra 1 e 344690")
        else:
            news = topnews.iloc[c]

            print(news['title'])
            print(news['link'])
            print(news['category'])


def main():
    run = True

    while run:

        print("Cosa vuoi fare?")
        print("1) Valuta classificatore")
        print("2) Classifica una notizia (LINK)")
        print("3) Classifica il dataset di notizie d'ultim'ora di ANSA")
        print("4) Classifica una stringa data in input")
        print("0) Esci")

        try:
            c = int(input())
        except ValueError:
            c = -1

        try:
            if c == 0:
                run = False
            elif c == 1:
                valutazione(classificate, stopwords)
            elif c == 2:
                classify()
            elif c == 3:
                classify_ansa()
            elif c == 4:
                print(classifier.predict(vectorizer.transform([input('Scrivi testo da classificare: ')])))
            else:
                print('Comando non valido')
        except Exception as e:
            print('errore')
            print(e)


    print('Fine esecuzione')


if __name__ == '__main__':
    main()

