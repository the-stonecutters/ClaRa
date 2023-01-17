from sklearn.naive_bayes import ComplementNB
import numpy as np

from iperparametri.validation_curve import do_validation_curve
import matplotlib.pyplot as plt


for param_name, param_range, logspace in (
    ('alpha', np.logspace(-7, 1, 7), True),
):
    do_validation_curve(ComplementNB(), param_name, param_range, logspace)
plt.show()