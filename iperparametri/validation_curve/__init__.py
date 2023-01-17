import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

from iperparametri import load_XY


def plot_validation_curve(train_scores, test_scores, param_range, parameter, cln, logspace=False):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    fig.title = cln+"_"+parameter
    ax.set_title("Validation Curve with "+cln)
    ax.set_xlabel(parameter)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.1)
    lw = 2
    if logspace:
        ax.semilogx(
            param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
        )
    else:
        ax.plot(
            param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
        )
    ax.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    ax.plot(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    ax.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    ax.legend(loc="best")
    #plt.show()


def do_validation_curve(alg, param_name, param_range, logspace=False):
    X, Y = load_XY()

    train_scores, test_scores = validation_curve(
        alg,
        X,
        Y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    plot_validation_curve(train_scores, test_scores, param_range, param_name, alg.__class__.__name__, logspace)
    return train_scores, test_scores

