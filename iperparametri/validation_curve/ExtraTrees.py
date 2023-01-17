from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

from iperparametri.validation_curve import do_validation_curve
import matplotlib.pyplot as plt


for param_name, param_range in (
    #('n_estimators', np.arange(1,200)),
    #('max_depth', np.arange(45,150)),
    #('min_samples_split', np.arange(15,50))
    #('min_samples_leaf', np.arange(1,30)),
    ('max_features', np.arange(50,70)),
):
    do_validation_curve(ExtraTreesClassifier(), param_name, param_range)
    plt.show()