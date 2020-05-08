from typing import Tuple

import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler

LEVELS = ["time", "lat", "lon"]

# @task # get reference cube
def get_reference_cube(data: xr.DataArray) -> pd.DataFrame:
    """Wrapper Function to get reference cube"""
    return data.to_dataframe().dropna().reorder_levels(LEVELS)


def get_common_indices(
    reference_df: pd.DataFrame, density_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = density_df.index.intersection(reference_df.index)
    return reference_df.loc[idx, :], density_df.loc[idx, :]


def standardizer_data(
    X: pd.DataFrame, Y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize the data"""
    normalizer = StandardScaler(with_mean=True, with_std=True)

    # standardize X values
    X = normalizer.fit_transform(X)
    # X = pd.DataFrame(data=X_values, index=X.index, columns=X.columns)

    # standardize Y Values
    Y = normalizer.fit_transform(Y)
    # Y = pd.DataFrame(data=Y_values, index=Y.index, columns=Y.columns)

    return X, Y


# ----------------------------------------------------------
# Matching Temporal Resolutions
# ----------------------------------------------------------

# TODO: Check TommyLee Scripts
# https://github.com/tommylees112/esowc_notes/blob/master/src/preprocessing_utils.py
# TODO: Get Union TimeSlice
