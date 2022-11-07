from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import nltk

nltk.download('stopwords')
stemmer = nltk.stem.SnowballStemmer('italian')
stopwords = nltk.corpus.stopwords.words('italian')
stopwords.extend(['ansa'])

classificate = pd.read_csv('news.csv').drop(columns=['created_at', 'updated_at'])
classificate = classificate[classificate['news_body'].notnull()]
len = classificate['news_id'].size

vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3))
X = vectorizer.fit_transform(classificate['news_body'])


# Carico la lista di preferenze utenti
preferenze = pd.read_csv('preferences.csv')

# ID di un utente che ha espresso alcune preferenze
user_id = 1154626546
user_id = 1141519299
# Lista di notizie votate
voted = preferenze[preferenze['user_id'] == user_id]
voted_ids = np.asarray(preferenze['news_id'])

voted_vec = vectorizer.transform(voted['news_body'])

# 0 = Non interessante, 1 = interessante

# Utilizziamo il classificatore di Rocchio per creare
# due centroidi, uno per le notizie preferite e uno per quelle sfavorite
print('metodo rocchio')

NC = NearestCentroid()

NC.fit(voted_vec, voted['preference'])

if NC.classes_[0] == 0:
    centroids = NC.centroids_
else:
    centroids = [NC.centroids_[1], NC.centroids_[0]]


BETA = 0.8
GAMMA = (1-BETA)

distance = list(enumerate(cosine_similarity(centroids, X)))
scores = []

for i in range(0, len):
    news_id = classificate.iloc[i]['news_id']
    # Teniamo in considerazione solo le notizie non votate
    if news_id not in voted_ids:
        # La metrica finale terrà in cosiderazione sia i feedback positivi che quelli negativi
        score = distance[1][1][i] * BETA - distance[0][1][i] * GAMMA
        scores.append([i, score])

scores_sorted = sorted(scores, reverse=True, key=lambda x: x[1])

best_correlation = [x[0] for x in scores_sorted[:10]]

print('Raccomandazioni: ')

for v in best_correlation:
    news = classificate.iloc[v]

    print(news['news_id'], ' - ' + news['news_title'] + ' - ' + news['news_link'])


print('metodo kmeans')


n_clusters = 30
preferiti = preferenze[preferenze['preference'] == 1].copy()
KM = KMeans(n_clusters=n_clusters, random_state=42)
KM.fit(vectorizer.transform(preferiti['news_body']))
preferiti['cluster'] = KM.labels_

pref_utente = preferiti[preferiti['user_id'] == user_id]
cluster_count = pref_utente['cluster'].value_counts()
sum = cluster_count.sum()

distance = list(enumerate(cosine_similarity(KM.cluster_centers_, X)))
scores = []

for i in range(0, len):
    news_id = classificate.iloc[i]['news_id']
    # Teniamo in considerazione solo le notizie non votate
    if news_id not in voted_ids:
        score = 0
        for c, v in cluster_count.items():
            score = score + distance[c][1][i] * v

        scores.append([i, score/sum])

scores_sorted = sorted(scores, reverse=True, key=lambda x: x[1])[:10]

best_correlation = [x[0] for x in scores_sorted]

print('Raccomandazioni: ')

for v in best_correlation:
    news = classificate.iloc[v]

    print(news['news_id'], ' - ' + news['news_title'] + ' - ' + news['news_link'])

