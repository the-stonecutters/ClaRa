import pandas as pd
import nltk
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer

from recommender.rocchio import RocchioRecommender

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('italian')
stopwords.extend(['ansa'])

classificate = pd.read_csv('news.csv').drop(columns=['created_at', 'updated_at'])
classificate = classificate[classificate['news_body'].notnull()]
len = classificate['news_id'].size

vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3), max_features=50000)
X = vectorizer.fit_transform(classificate['news_body'])


# Carico la lista di preferenze utenti
preferenze = pd.read_csv('preferences.csv')

# ID di un utente che ha espresso alcune preferenze
user_id = 1154626546
#user_id = 1141519299

# Lista di notizie votate
voted = preferenze[preferenze['user_id'] == user_id].reset_index(drop=True)
voted_ids = np.asarray(preferenze['news_id'])

voted_vec = vectorizer.transform(voted['news_body'])
voted_pref = np.asarray(voted['preference'])
# 0 = Non interessante, 1 = interessante

# Utilizziamo il classificatore di Rocchio per creare
# due centroidi, uno per le notizie preferite e uno per quelle sfavorite
print('metodo rocchio')

BETA = 0.8
rocchio = RocchioRecommender(beta=BETA)
rocchio.fit(voted_vec, voted['preference'])


raccomandazioni = classificate.copy()
raccomandazioni['score'] = rocchio.predict_proba(X)

raccomandazioni = raccomandazioni.sort_values('score', ascending=False)

i = 0
for _, news in raccomandazioni.iterrows():
    if not news['news_id'] in voted_ids:
       i = i+1
       print(news['score'], news['news_id'], ' - ' + news['news_title'] + ' - ' + news['news_link'])
    if i == 10:
        break

