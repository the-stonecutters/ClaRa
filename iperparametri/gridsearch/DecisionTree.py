from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from iperparametri import load_XY

X, Y = load_XY()


kf = StratifiedKFold()

parameters = {
    'min_samples_split': [1, 2, 3, 4, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [10, 20, 30, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid=parameters,
    cv=kf,
    n_jobs=-1,
    scoring='f1_macro',
    verbose=2
)

grid_search.fit(X, Y)
print(grid_search.best_params_)
print(grid_search.best_score_)
