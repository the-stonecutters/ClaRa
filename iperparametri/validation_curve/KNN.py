from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from iperparametri.validation_curve import do_validation_curve
import matplotlib.pyplot as plt


for param_name, param_range in (
    ('n_neighbors', np.arange(1, 15)),
    ('leaf_size', np.arange(1, 50)),
    ('p', np.arange(1, 10)),
):
    do_validation_curve(KNeighborsClassifier(), param_name, param_range)
plt.show()