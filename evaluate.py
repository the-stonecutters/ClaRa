from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

import matplotlib.pyplot as plt
import numpy as np


def evaluate(Ci, X_train, X_test, Y_train, Y_test, Cn=None):
    Ci.fit(X_train, Y_train)
    pred = Ci.predict(X_test)

    result = {
        'name': Cn,
        'pred': pred,
        'test': Y_test,
        'accuracy': accuracy_score(Y_test, pred),
        'precision': precision_score(Y_test, pred, average='macro'),
        'recall': recall_score(Y_test, pred, average='macro'),
        'f1_score': f1_score(Y_test, pred, average='macro')
    }

    return result


def confusion(result, labels=None, name=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(result['test'], result['pred'], ax=ax)
    if labels is not None:
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
    if name is None and result['name'] is not None:
        name = result['name']
    if name is not None:
        _ = ax.set_title(
            f"Confusion Matrix for {name}"
        )

    plt.show()


def valutazione(classificate, stopwords):
    classificate['news_category'].value_counts().plot(kind='bar', ylabel='#')
    plt.show()

    data_train, data_test = train_test_split(classificate, test_size=0.33, random_state=42)

    target_names = np.asarray(data_train['news_category'].unique())

    vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(data_train['news_body'])
    X_test = vectorizer.transform(data_test['news_body'])
    Y_train = data_train['news_category']
    Y_test = data_test['news_category']

    classifier = ComplementNB(alpha=0.2136)

    name = classifier.__class__.__name__

    result = evaluate(classifier, X_train, X_test, Y_train, Y_test, name)
    confusion(result, target_names)

