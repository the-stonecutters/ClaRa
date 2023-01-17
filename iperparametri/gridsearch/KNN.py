from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np

from iperparametri import load_XY

X, Y = load_XY()


kf = StratifiedKFold()

parameters = {
    'n_neighbors': [5, 10, 13],
    'leaf_size': [5, 15, 25, 35, 45],
    'p': [1, 2],
    'weights': ['uniform', 'distance'],
    'algorithm': ['brute'],
    'metric': ['cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean'],
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=parameters,
    cv=kf,
    n_jobs=-1,
    scoring='f1_macro',
    error_score=np.nan,
    verbose=2
)

grid_search.fit(X, Y)
print(grid_search.best_params_)
print(grid_search.best_score_)
