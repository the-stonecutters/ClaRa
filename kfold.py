from sklearn.model_selection import StratifiedKFold
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

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('italian')
stopwords.extend(['ansa'])

classificate = pd.read_csv('news.csv')
classificate = classificate[classificate['news_body'].notnull()]

classificate['news_category'].value_counts().plot(kind='bar', ylabel='#')

vectorizer = TfidfVectorizer(max_df=0.5, use_idf=True, stop_words=stopwords, ngram_range=(1, 3))

X = vectorizer.fit_transform(classificate['news_body'])
Y = classificate['news_category']

kf = StratifiedKFold()
f = open('results.csv', 'w')
f.write('classificatore,n_fold,precision,recall,accuracy,f1_score\n')

mediaFinale = {
    'name': [],
    'avg': [],
    'error': []
}
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
    '''
    dfAvg = pd.DataFrame([{
        'test_accuracy': df['test_accuracy'].mean(),
        'test_precision_macro': df['test_precision_macro'].mean(),
        'test_recall_macro': df['test_recall_macro'].mean(),
        'test_f1_macro': df['test_f1_macro'].mean()
    }])
    '''
    for i in range(0, 5):
        f.write(
            "{},{},{},{},{},{}\n".format(name, i, df.iloc[i]['test_precision_macro'], df.iloc[i]['test_recall_macro'],
                                         df.iloc[i]['test_accuracy'], df.iloc[i]['test_f1_macro']))
    dfAvg = df.mean()
    dfVar = df.var()
    dfStd = df.std()
    # df = pd.concat([df, dfAvg], ignore_index=True)
    # df = df.append(dfAvg, ignore_index=True)
    # df = df.reset_index([1, 2, 3, 4, 5, 'avg'])
    print(df)
    print('Media')
    print(dfAvg)
    print('Scostamento')
    print(dfStd)

    print("F1 avg: ", dfAvg['test_f1_macro'], dfStd['test_f1_macro'])
    _, ax = plt.subplots()
    df.plot(kind='bar', ax=ax, title=name)
    ax.set_xticklabels([1, 2, 3, 4, 5])
    # _.savefig(name+'.png')
    # _.savefig(name+'.svg')
    _, ax = plt.subplots()
    dfAvg.plot(kind='barh', ax=ax, title=name + ' AVG', xerr=dfStd)
    # _.savefig(name+' AVG.png')
    # _.savefig(name+' AVG.svg')
    # ax.set_xticklabels(['avg'])
    print("\n")

    mediaFinale['name'].append(name)
    mediaFinale['avg'].append(dfAvg['test_f1_macro'])
    mediaFinale['error'].append(dfStd['test_f1_macro'])

_, ax = plt.subplots()
ax.barh(mediaFinale['name'], mediaFinale['avg'], xerr=mediaFinale['error'])
ax.set_title('Media F1 per ogni classificatore')
plt.show()
