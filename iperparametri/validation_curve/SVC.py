from sklearn.svm import SVC
import numpy as np

from iperparametri.validation_curve import do_validation_curve
import matplotlib.pyplot as plt


for param_name, param_range in (
    ('tol', np.logspace(-7, 0, 7)),
    ('C', np.arange(0, 2, 0.1)),
):
    do_validation_curve(SVC(), param_name, param_range)
plt.show()