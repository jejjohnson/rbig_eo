from typing import Dict, Optional

import pandas as pd
from src.models.similarity import rv_coefficient, rbig_it_measures, cka_coefficient
from src.models.baseline import train_rf_model, train_rf_model, train_ridge_lr_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.features import Metrics
import numpy as np
from sklearn.utils import check_random_state


def get_similarity_scores(
    X_ref: pd.DataFrame,
    Y_compare: pd.DataFrame,
    smoke_test: bool = False,
    subsample: Optional[int] = 10_000,
) -> Dict:

    if smoke_test is True:
        subsample = 100

    # RV Coefficient
    rv_results = rv_coefficient(X_ref, Y_compare, subsample=subsample)

    # CKA Coefficient
    cka_results = cka_coefficient(X_ref, Y_compare, subsample=subsample)

    # RBIG Coefficient
    rbig_results = rbig_it_measures(X_ref, Y_compare, subsample=subsample * 10)

    results = {
        **rv_results,
        **cka_results,
        **rbig_results,
    }

    return results


def get_regression_models(X: np.ndarray, y: np.ndarray, subsample: int = 10_000):

    subsample = np.minimum(X.shape[0], subsample)

    if subsample is not None:
        rng = check_random_state(123)
        X = rng.permutation(X)[:subsample, :]
        y = rng.permutation(y)[:subsample, :]
    random_state = 123

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.8, random_state=random_state
    )

    # normalize inputs
    x_normalizer = StandardScaler(with_mean=True, with_std=False)

    xtrain_norm = x_normalizer.fit_transform(xtrain)
    xtest_norm = x_normalizer.transform(xtest)

    # remove mean outputs
    y_normalizer = StandardScaler(with_std=False)

    ytrain_norm = y_normalizer.fit_transform(ytrain)
    ytest_norm = y_normalizer.transform(ytest)

    # linear regresion model
    rlr_model = train_ridge_lr_model(xtrain_norm, ytrain_norm)
    ypred = rlr_model.predict(xtest_norm)
    # get statistics
    rlr_metrics = Metrics().get_all(ypred, ytest_norm, "rlr")

    # RF Model
    rf_model = train_rf_model(xtrain_norm, ytrain_norm)
    ypred = rf_model.predict(xtest_norm)
    # get statistics
    rf_metrics = Metrics().get_all(ypred, ytest_norm, "rf")

    results = {**rlr_metrics, **rf_metrics}
    return results
