from sklearn.linear_model import SGDClassifier
import numpy as np

from iperparametri.validation_curve import do_validation_curve
import matplotlib.pyplot as plt


for param_name, param_range, logspace in (
    ('alpha', np.logspace(-7, 1, 7), True),
    ('l1_ratio', np.arange(0, 1, 0.05), False),
    ('tol', np.logspace(-7, 1, 7), True),
    ('validation_fraction', np.arange(0, 1, 0.1), False),
):
    do_validation_curve(SGDClassifier(), param_name, param_range, logspace)
plt.show()