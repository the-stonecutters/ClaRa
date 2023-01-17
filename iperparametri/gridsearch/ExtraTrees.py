from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from iperparametri import load_XY

X, Y = load_XY()


kf = StratifiedKFold()

parameters = {
    'n_estimators': [50, 100, 150],
    'min_samples_split': [15, 20],
    'min_samples_leaf': [1, 5],
    'max_features': [40, 60, 'sqrt', 'log2'],
    'max_depth': [45, 150],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid_search = GridSearchCV(
    ExtraTreesClassifier(),
    param_grid=parameters,
    cv=kf,
    n_jobs=-1,
    scoring='f1_macro',
    verbose=2
)

grid_search.fit(X, Y)
print(grid_search.best_params_)
print(grid_search.best_score_)