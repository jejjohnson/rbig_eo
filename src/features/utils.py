from typing import Tuple, Optional
from sklearn.utils import check_random_state
import numpy as np
import pandas as pd


def move_variables(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    #     cond1 = df['variable1'] == variable
    cond = df["variable2"] == variable
    df.loc[cond, ["variable2", "variable1"]] = df.loc[
        cond, ["variable1", "variable2"], ["rbig_H_x", "rbig_H_y"]
    ].values

    return df


def subset_data(
    X: np.ndarray, subsample: Optional[int] = None, random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    
    idx = subset_indices(X, subsample, random_state)

    return X[subset_indices, :]

def subset_indices(
    X: np.ndarray, subsample: Optional[int] = None, random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:

    if subsample is not None and subsample < X.shape[0]:
        rng = check_random_state(random_state)
        indices = np.arange(X.shape[0])
        subset_indices = rng.permutation(indices)[:subsample]
        return subset_indices
    else:
        return None