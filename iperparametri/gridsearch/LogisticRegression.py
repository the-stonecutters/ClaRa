from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np

from iperparametri import load_XY

X, Y = load_XY()


kf = StratifiedKFold()

parameters = {
    'tol': [1e-7, 1e-4, 1e-2, 1],
    'C': [1.5, 2.0, 2.5, 3.0, 3.5],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

grid_search = GridSearchCV(
    LogisticRegression(),
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
