from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
import pandas as pd
import nltk
nltk.download('stopwords')

stemmer = nltk.stem.SnowballStemmer('italian')
stopwords = nltk.corpus.stopwords.words('italian')
stopwords.extend(['ansa'])

classificate = pd.read_csv('news.csv')
classificate = classificate[classificate['news_body'].notnull()]

#classificate['news_category'].value_counts().plot(kind='bar', ylabel='#')

vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3))

X = vectorizer.fit_transform(classificate['news_body'])
Y = classificate['news_category']

kf = StratifiedKFold()

for classifier, name in (
            (ComplementNB(alpha=0.23), 'Naive Bayes'),
            (SGDClassifier(alpha=1e-5, tol=0.1), 'SGD Classifier'),
            #(RidgeClassifier(), 'Ridge Classifier'),
            #(KNeighborsClassifier(), 'Neirest Neighbors'),
            #(RandomForestClassifier(), 'Random Forest'),
    ):
    cv_model = cross_validate(
        classifier,
        X,
        Y,
        cv=kf,
        n_jobs=-1,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        return_estimator=True
    )

    df = pd.DataFrame(cv_model)
    print("Classificatore: " + name)
    print("Tempo di training medio: ", df['fit_time'].mean())
    df = df.drop(columns=['fit_time', 'score_time', 'estimator'])

    dfAvg = pd.DataFrame([{
        'test_accuracy': df['test_accuracy'].mean(),
        'test_precision_macro': df['test_precision_macro'].mean(),
        'test_recall_macro': df['test_recall_macro'].mean(),
        'test_f1_macro': df['test_f1_macro'].mean()
    }])

    df = pd.concat([df, dfAvg], ignore_index=True)
    print(df)

    print("F1 avg: ", df['test_f1_macro'].mean())
    df.plot(kind='bar', title=name)
    plt.show()
