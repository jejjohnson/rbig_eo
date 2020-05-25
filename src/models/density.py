from rbig.rbig import RBIGMI, RBIG as OLDRBIG
from rbig.model import RBIG
from typing import Dict, Optional
import time
import numpy as np
from src.models.utils import subset_indices


def get_rbig_model(
    X: np.ndarray,
    params: Optional[Dict] = None,
    random_state: int = 123,
    verbose: bool = False,
    method: str = "old",
) -> None:

    if method == "old":
        rbig_model = _get_rbig_model_old(
            X, params=params, random_state=random_state, verbose=verbose,
        )
    elif method == "new":
        rbig_model = _get_rbig_model_new(
            X, params=params, random_state=random_state, verbose=verbose,
        )
    else:
        raise ValueError(f"Unrecognized method: {method}")

    return rbig_model


def _get_rbig_model_old(
    X: np.ndarray,
    params: Optional[Dict] = None,
    random_state: int = 123,
    verbose: bool = False,
) -> None:

    n_layers = 10000
    rotation_type = "PCA"
    random_state = 0
    zero_tolerance = 60
    pdf_extension = 10

    # Initialize RBIG class
    rbig_model = OLDRBIG(
        n_layers=n_layers,
        rotation_type=rotation_type,
        random_state=random_state,
        pdf_extension=pdf_extension,
        zero_tolerance=zero_tolerance,
        verbose=verbose,
    )

    return rbig_model


def _get_rbig_model_new(
    X: np.ndarray,
    params: Optional[Dict] = None,
    random_state: int = 123,
    verbose: bool = False,
    method: str = "old",
) -> None:

    n_layers = 10_000
    rotation_type = "PCA"
    random_state = 0
    zero_tolerance = 60
    pdf_extension = 10
    tolerance = None
    method = "kdefft"
    n_quantiles = 50
    bw_method = "scott"
    # Initialize RBIG class
    rbig_model = RBIG(
        n_layers=n_layers,
        rotation_type=rotation_type,
        random_state=random_state,
        zero_tolerance=zero_tolerance,
        tolerance=tolerance,
        pdf_extension=pdf_extension,
        verbose=0,
        method=method,
        n_quantiles=n_quantiles,
    )

    return rbig_model
