import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
import pandas as pd
import nltk

from evaluate import evaluate, confusion

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('italian')
stopwords.extend(['ansa'])

classificate = pd.read_csv('news.csv')
classificate = classificate[classificate['news_body'].notnull()]

classificate['news_category'].value_counts().plot(kind='bar', ylabel='#')

for classifier, name in (
        (RidgeClassifier(alpha=0.01, tol=0.1, solver='sparse_cg', random_state=42), 'Ridge Classifier'),
        (ComplementNB(alpha=0.2136), 'Naive Bayes'),
        (SGDClassifier(alpha=1e-06, tol=0.1, loss='hinge', penalty='l2', random_state=42), 'SGD Classifier'),
        (KNeighborsClassifier(algorithm='brute', leaf_size=5, n_neighbors=10, p=1, weights='distance', metric='cosine'),
         'Neirest Neighbors'),
        (ExtraTreesClassifier(criterion='gini', max_depth=150, max_features='sqrt', min_samples_leaf=1,
                              min_samples_split=20, n_estimators=150, random_state=42), 'Extra Trees'),
        (RandomForestClassifier(criterion='gini', max_depth=100, max_features='sqrt', max_samples=600,
                                min_samples_leaf=2, min_samples_split=4, n_estimators=220, random_state=42),
         'Random Forest'),
        (DecisionTreeClassifier(criterion='gini', max_features='sqrt', min_samples_leaf=1, min_samples_split=3,
                                random_state=42), 'Decision Tree'),
        (SVC(C=1.75, decision_function_shape='ovo', gamma='scale', kernel='sigmoid', tol=1.2, random_state=42),
         'SVC Classifier'),
        (LogisticRegression(C=3.5, penalty='l1', solver='liblinear', tol=1e-07, random_state=42), 'Logistic Regression')
):
    data_train, data_test = train_test_split(classificate, test_size=0.33, random_state=42)
    target_names = np.asarray(data_train['news_category'].unique())

    vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(data_train['news_body'])
    X_test = vectorizer.transform(data_test['news_body'])
    Y_train = data_train['news_category']
    Y_test = data_test['news_category']

    result = evaluate(classifier, X_train, X_test, Y_train, Y_test, name)
    confusion(result, target_names)

