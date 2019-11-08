from .entropy import KNNEstimator
from sklearn.base import BaseEstimator
from .ensemble import Batch
from typing import Optional
import numpy as np
from sklearn.utils import check_array

import sys

sys.path.insert(0, "/home/emmanuel/code/rbig")
from rbig import RBIG, RBIGMI


class MutualInformation(BaseEstimator):
    def __init__(self, estimator: str = "knn", kwargs: Optional[dict] = None) -> None:
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Data to be estimated.
        """
        X = check_array(X)
        if self.estimator == "knn":
            self.model = (
                KNNEstimator(**self.kwargs)
                if self.kwargs is not None
                else KNNEstimator()
            )
        elif self.estimator in ["rbig", "kde", "histogram"]:
            raise NotImplementedError(f"{self.estimator} is not implemented yet.")

        else:
            raise ValueError(f"Unrecognized estimator: {self.estimator}")
        if Y is not None:
            Y = check_array(Y)
            self._fit_mutual_info(X, Y)
        else:
            raise ValueError(f"X dims are less than 2. ")

        return self

    def _fit_multi_info(self, X: np.ndarray) -> float:

        # fit full
        model_full = self.model.fit(X)
        H_x = model_full.score(X)

        # fit marginals
        H_x_marg = 0
        for ifeature in X.T:

            model_marginal = self.model.fit(ifeature)
            H_x_marg += model_marginal.score(ifeature)

        # calcualte the multiinformation
        self.MI = H_x_marg - H_x

        return H_x_marg - H_x

    def _fit_mutual_info(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:

        # MI for X
        model_x = self.model.fit(X)
        H_x = model_x.score(X)
        print("Marginal:", H_x)

        # MI for Y
        model_y = self.model.fit(Y)
        H_y = model_y.score(Y)
        print("Marginal:", H_y)

        # MI for XY
        model_xy = self.model.fit(np.hstack([X, Y]))
        H_xy = model_xy.score(X)
        print("Full:", H_xy)

        # save the MI
        self.MI = H_x + H_y - H_xy

        return self.MI

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        return self.MI


class TotalCorrelation(BaseEstimator):
    def __init__(self, estimator: str = "knn", kwargs: Optional[dict] = None) -> None:
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Data to be estimated.
        """
        X = check_array(X)

        if self.estimator == "knn":
            self.model = (
                KNNEstimator(**self.kwargs)
                if self.kwargs is not None
                else KNNEstimator()
            )
        elif self.estimator in ["rbig", "kde", "histogram"]:
            raise NotImplementedError(f"{self.estimator} is not implemented yet.")

        else:
            raise ValueError(f"Unrecognized estimator: {self.estimator}")

        if y is None and X.shape[1] > 1:

            self._fit_multi_info(X)
        else:
            raise ValueError(f"X dims are less than 2. ")

        return self

    def _fit_multi_info(self, X: np.ndarray) -> float:

        # fit full
        model_full = self.model.fit(X)
        H_x = model_full.score(X)
        print("Full:", H_x)
        # fit marginals
        H_x_marg = 0
        for ifeature in X.T:

            model_marginal = self.model.fit(ifeature[:, None])

            H_xi = model_marginal.score(ifeature[:, None])
            print("Marginal:", H_xi)
            H_x_marg += H_xi

        # calcualte the multiinformation
        self.MI = H_x_marg - H_x

        return self

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        return self.MI


class RBIGEstimator(BaseEstimator, Batch):
    def __init__(
        self,
        n_layers: int = 10_000,
        rotation_type: str = "PCA",
        zero_tolerance: int = 60,
        pdf_extension: int = 10,
        pdf_resolution: Optional[int] = None,
        tolerance: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ):
        # Initialize super class
        Batch.__init__(self, batch_size=batch_size, random_state=random_state)

        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.zero_tolerance = zero_tolerance
        self.tolerance = tolerance
        self.pdf_extension = pdf_extension
        self.pdf_resolution = pdf_resolution
        self.verbose = verbose
        self.shuffle = shuffle

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Data to be estimated.
        """

        # Case I - Mutual Information
        if self.batch_size is not None:
            self.mi_ = self._fit_batches(X, y)
        else:
            self.mi_ = self._fit(X, y)

        return self

    def _fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> float:

        X = check_array(X, ensure_2d=True)

        # Case I - Mutual Information
        if Y is None:
            return self._fit_total_correlation(X)
        # Case II - Total Correlation
        else:

            Y = check_array(Y, ensure_2d=True)
            return self._fit_mutual_info(X, Y)

    def _fit_mutual_info(self, X: np.ndarray, Y: np.ndarray) -> float:

        # initialize rbig model
        rbig_mi_model = RBIGMI(
            n_layers=self.n_layers,
            rotation_type=self.rotation_type,
            random_state=self.random_state,
            zero_tolerance=self.zero_tolerance,
            tolerance=self.tolerance,
            pdf_extension=self.pdf_extension,
            pdf_resolution=self.pdf_resolution,
            verbose=self.verbose,
        )

        # fit RBIG model to data
        rbig_mi_model.fit(X, Y)

        # return mutual info
        return rbig_mi_model.mutual_information() * np.log(2)

    def _fit_total_correlation(self, X: np.ndarray) -> float:

        # 1. Calculate the K-nearest neighbors
        rbig_model = RBIG(
            n_layers=self.n_layers,
            rotation_type=self.rotation_type,
            random_state=self.random_state,
            zero_tolerance=self.zero_tolerance,
            tolerance=self.tolerance,
            pdf_extension=self.pdf_extension,
            pdf_resolution=self.pdf_resolution,
            verbose=self.verbose,
        )

        rbig_model.fit(X)

        # estimation
        return rbig_model.mutual_information * np.log(2)

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        return self.mi_

