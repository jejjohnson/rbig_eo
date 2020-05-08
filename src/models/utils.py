from typing import Tuple, Optional
from sklearn.utils import check_random_state
import numpy as np


def subset_indices(
    X: np.ndarray,
    Y: np.ndarray,
    subsample: Optional[int] = None,
    random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:

    if subsample is not None and subsample < X.shape[0]:
        rng = check_random_state(random_state)
        indices = np.arange(X.shape[0])
        subset_indices = rng.permutation(indices)[:subsample]
        X = X[subset_indices, :]
        Y = Y[subset_indices, :]

    return X, Y
