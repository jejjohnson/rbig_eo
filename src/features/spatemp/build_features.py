import xarray as xr
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def select_region(
    ds: Union[xr.DataArray, xr.Dataset], region: str = "europe"
) -> Union[xr.DataArray, xr.Dataset]:

    if region == "europe":
        return ds.sel(lat=slice(71.5, 35.5), lon=slice(-18.0, 60.0))
    else:
        raise ValueError("Unrecognized region:", region)


def normalize_inputs(
    Xtrain: Union[np.ndarray, pd.DataFrame], Xtest: Union[np.ndarray, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:

    # normalize inputs
    x_normalizer = StandardScaler(with_mean=True, with_std=False)

    xtrain_norm = x_normalizer.fit_transform(Xtrain)
    xtest_norm = x_normalizer.transform(Xtest)

    # # remove mean outputs
    # y_normalizer = StandardScaler(with_std=False)

    # ytrain_norm = y_normalizer.fit_transform(ytrain)
    # ytest_norm = y_normalizer.transform(ytest)

    return xtrain_norm, xtest_norm


def normalize_outputs(
    Ytrain: Union[np.ndarray, pd.DataFrame], Ytest: Union[np.ndarray, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:

    # remove mean outputs
    y_normalizer = StandardScaler(with_std=False)

    ytrain_norm = y_normalizer.fit_transform(Ytrain)
    ytest_norm = y_normalizer.transform(Ytest)

    return ytrain_norm, ytest_norm

