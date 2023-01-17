import pandas as pd
import nltk
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

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
voted = preferenze[preferenze['user_id'] == user_id]
voted_ids = np.asarray(preferenze['news_id'])

print('metodo kmeans')

preferiti = preferenze[preferenze['preference'] == 1].copy()
preferiti_vec = vectorizer.transform(preferiti['news_body'])

print('ricerca numero cluster ottimale')
inertias = {}
K = range(2, 50)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(preferiti_vec)
    inertias[k] = km.inertia_

kn = KneeLocator(x=list(inertias.keys()),
                 y=list(inertias.values()),
                 curve='convex',
                 direction='decreasing')

print(kn)
plt.plot(K, inertias.values(), 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

n_clusters = kn.knee
print(kn.elbow)
print(n_clusters)

KM = KMeans(n_clusters=n_clusters, random_state=42)
KM.fit(preferiti_vec)
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

