import numpy as np
from sklearn.linear_model import LogisticRegression

from iperparametri.validation_curve import do_validation_curve
import matplotlib.pyplot as plt


for param_name, param_range in (
    ('tol', np.logspace(-7, 0, 7)),
    ('C', np.arange(1, 4, 0.2)),
):
    do_validation_curve(LogisticRegression(), param_name, param_range)
plt.show()