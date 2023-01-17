from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np

from iperparametri import load_XY

X, Y = load_XY()


kf = StratifiedKFold()

parameters = {
    'tol': [0.6, 0.8, 1.0, 1.2],
    'C': [1.0, 1.25, 1.50, 1.75, 2.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'decision_function_shape': ['ovo', 'ovr']
}

grid_search = GridSearchCV(
    SVC(),
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
