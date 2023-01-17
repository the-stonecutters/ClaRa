from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

from iperparametri.validation_curve import do_validation_curve

param_range = np.arange(1, 50)

param_name = 'max_features'
do_validation_curve(RandomForestClassifier(), param_name, param_range)
plt.show()