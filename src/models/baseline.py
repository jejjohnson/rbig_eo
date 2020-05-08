from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.base import BaseEstimator
from typing import Optional, Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV


def train_ridge_lr_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
    n_jobs: int = 1,
) -> BaseEstimator:
    # Initialize GLM
    lr_model = RidgeCV()

    # train GLM
    t0 = time.time()
    lr_model.fit(xtrain, ytrain)
    t1 = time.time() - t0
    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return lr_model


def train_rf_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    params: Optional[Dict] = None,
) -> BaseEstimator:
    """Train a basic Random Forest (RF) Regressor 
    Parameters
    ----------
    xtrain : np.ndarray, pd.DataFrame 
             (n_samples x d_features)
             input training data
    
    ytrain : np.ndarray, pd.DataFrame 
             (n_samples x p_outputs)
             labeled training data 
    
    verbose : int, default=0
        option to print out training messages 
    Returns 
    -------
    rf_model : BaseEstimator
        the trained model
    """
    if params is None:
        params = {
            "n_estimators": 100,
            "criterion": "mse",
            "n_jobs": -1,
            "random_state": 123,
            "warm_start": False,
            "verbose": 0,
        }
    # initialize baseline RF model
    rf_model = RandomForestRegressor(**params)
    # train RF model
    t0 = time.time()
    rf_model.fit(xtrain, ytrain)
    t1 = time.time() - t0

    if params["verbose"] > 0:
        print(f"Training time: {t1:.3f} secs.")
    return rf_model
