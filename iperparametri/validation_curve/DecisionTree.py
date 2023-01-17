import numpy as np
from sklearn.tree import DecisionTreeClassifier

from iperparametri.validation_curve import do_validation_curve
import matplotlib.pyplot as plt


for param_name, param_range in (
    #('max_depth', np.arange(1, 100)),
    ('min_samples_split', np.arange(1, 5)),
    ('min_samples_leaf', np.arange(1, 5)),
    #('max_features', np.arange(1, 60))
):
    do_validation_curve(DecisionTreeClassifier(), param_name, param_range)
plt.show()