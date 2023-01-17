from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np

from iperparametri import load_XY

X, Y = load_XY()


kf = StratifiedKFold()

parameters = {
    'alpha': np.arange(0, 1, 0.0001),
}

grid_search = GridSearchCV(
    ComplementNB(),
    param_grid=parameters,
    cv=kf,
    n_jobs=-1,
    scoring='f1_macro',
    verbose=2
)

grid_search.fit(X, Y)
print(grid_search.best_params_)
print(grid_search.best_score_)