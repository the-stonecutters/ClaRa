from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from iperparametri import load_XY

X, Y = load_XY()


kf = StratifiedKFold()

parameters = {
    'n_estimators': [150, 200, 220],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_samples': [500, 600],
    'max_features': [25, 30, 'sqrt', 'log2'],
    'max_depth': [80, 100],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid=parameters,
    cv=kf,
    n_jobs=-1,
    scoring='f1_macro',
    verbose=2
)

grid_search.fit(X, Y)
print(grid_search.best_params_)
print(grid_search.best_score_)
