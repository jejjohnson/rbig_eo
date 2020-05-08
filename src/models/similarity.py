from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import check_random_state
from typing import Dict
from sklearn.preprocessing import KernelCenterer
from sklearn.gaussian_process.kernels import RBF
import time
from src.models.utils import subset_indices
from rbig.rbig import RBIGMI, RBIG


def rv_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
    subsample: Optional[int] = 10_000,
    random_state: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""
    t0 = time.time()
    X, Y = subset_indices(X, Y, subsample, random_state)

    # calculate the kernel matrices
    X_gram = linear_kernel(X)
    Y_gram = linear_kernel(Y)

    # center the kernels
    X_gramc = KernelCenterer().fit_transform(X_gram)
    Y_gramc = KernelCenterer().fit_transform(Y_gram)

    # normalizing coefficients (denomenator)
    x_norm = np.linalg.norm(X_gramc)
    y_norm = np.linalg.norm(Y_gramc)

    # frobenius norm of the cross terms (numerator)
    xy_norm = np.sum(X_gramc * Y_gramc)
    # rv coefficient
    pv_coeff = xy_norm / x_norm / y_norm

    return {
        "rv_coef": pv_coeff,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "xy_norm": xy_norm,
    }


def estimate_sigma(X: np.ndarray, percent: int = 50, heuristic: bool = False,) -> float:

    # get the squared euclidean distances

    kth_sample = int((percent / 100) * X.shape[0])
    dists = np.sort(squareform(pdist(X, "sqeuclidean")))[:, kth_sample]

    sigma = np.median(dists)

    if heuristic:
        sigma = np.sqrt(sigma / 2)
    return sigma


def cka_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
    subsample: Optional[int] = 10_000,
    random_state: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""

    X, Y = subset_indices(X, Y, subsample, random_state)

    # estimate sigmas
    sigma_X = estimate_sigma(X, percent=50)
    sigma_Y = estimate_sigma(Y, percent=50)

    # calculate the kernel matrices
    X_gram = RBF(sigma_X)(X)
    Y_gram = RBF(sigma_Y)(Y)

    # center the kernels
    X_gram = KernelCenterer().fit_transform(X_gram)
    Y_gram = KernelCenterer().fit_transform(Y_gram)

    # normalizing coefficients (denomenator)
    x_norm = np.linalg.norm(X_gram)
    y_norm = np.linalg.norm(Y_gram)

    # frobenius norm of the cross terms (numerator)
    xy_norm = np.sum(X_gram * Y_gram)
    # rv coefficient
    pv_coeff = xy_norm / x_norm / y_norm

    return {
        "cka_coeff": pv_coeff,
        "cka_y_norm": y_norm,
        "cka_x_norm": x_norm,
        "cka_xy_norm": xy_norm,
    }


def rbig_it_measures(
    X: np.ndarray,
    Y: np.ndarray,
    subsample: Optional[int] = 100_000,
    random_state: int = 123,
) -> Dict:
    X, Y = subset_indices(X, Y, subsample, random_state)
    n_layers = 10000
    rotation_type = "PCA"
    random_state = 0
    zero_tolerance = 60
    pdf_extension = 10

    rbig_results = {}

    t0 = time.time()
    # Initialize RBIG class
    H_rbig_model = RBIG(
        n_layers=n_layers,
        rotation_type=rotation_type,
        random_state=random_state,
        pdf_extension=pdf_extension,
        zero_tolerance=zero_tolerance,
    )

    # fit model to the data
    rbig_results["rbig_H_x"] = H_rbig_model.fit(X).entropy(correction=True)

    rbig_results["rbig_H_y"] = H_rbig_model.fit(Y).entropy(correction=True)
    rbig_results["rbig_H_time"] = time.time() - t0

    # Initialize RBIG class
    I_rbig_model = RBIGMI(
        n_layers=n_layers,
        rotation_type=rotation_type,
        random_state=random_state,
        pdf_extension=pdf_extension,
        zero_tolerance=zero_tolerance,
    )

    # fit model to the data
    t0 = time.time()
    rbig_results["rbig_I_xy"] = I_rbig_model.fit(X, Y).mutual_information()
    rbig_results["rbig_I_time"] = time.time() - t0

    t0 = time.time()
    rbig_results["rbig_I_xx"] = I_rbig_model.fit(X, X).mutual_information()
    rbig_results["rbig_Ixx_time"] = time.time() - t0

    # # calculate the variation of information coefficient
    # rbig_results["rbig_vi_coeff"] = variation_of_info(
    #     rbig_results["rbig_H_x"], rbig_results["rbig_H_y"], rbig_results["rbig_I_xy"]
    # )
    return rbig_results


def variation_of_info(H_X, H_Y, I_XY):
    return I_XY / np.sqrt(H_X) / np.sqrt(H_Y)
