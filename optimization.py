from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, validation_curve
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk



def plot_validation_curve(train_scores, test_scores, param_range, parameter, cln):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with "+cln)
    plt.xlabel(parameter)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()


nltk.download('stopwords')
stemmer = nltk.stem.SnowballStemmer('italian')
stopwords = nltk.corpus.stopwords.words('italian')
stopwords.extend(['ansa'])

classificate = pd.read_csv('news.csv')
classificate = classificate[classificate['news_body'].notnull()]

vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3))

X = vectorizer.fit_transform(classificate['news_body'])
Y = classificate['news_category']

kf = StratifiedKFold()


param_range = np.logspace(-7,0,4)
train_scores, test_scores = validation_curve(
    SGDClassifier(),
    X,
    Y,
    param_name="alpha",
    param_range=param_range,
    cv=kf,
    scoring="f1_macro",
    n_jobs=4
)

plot_validation_curve(train_scores, test_scores, param_range, 'alpha', 'SGD Classifier')


parameters = {
    'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
}

grid_search = GridSearchCV(
    SGDClassifier(),
    param_grid=parameters,
    cv=kf,
    n_jobs=-4,
    scoring='f1_macro',
)

grid_search.fit(X, Y)
print(grid_search.best_params_)
print(grid_search.best_score_)

