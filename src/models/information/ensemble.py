import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import gen_batches
from sklearn.model_selection import train_test_split
from typing import Optional
from sklearn.utils import shuffle


class Ensemble:
    def __init__(self):
        pass

    def _fit(self, X: np.ndarray) -> BaseEstimator:
        pass

    def _fit_ensemble(self, X: np.ndarray, n_models: int = 10) -> float:
        raise NotImplemented


class Batch:
    def __init__(self, random_state: int = 123):
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        IT method to fit to batches. Must be implemented by the user.
        """
        pass

    def _fit_batches(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None, batch_size: int = 1_000
    ) -> float:
        """
        Fits models to inherited class

        Parameters
        ----------
        X : np.ndarray
            The data to be fit.
        
        y : np.ndarray
            The second dataset to be fit

        batch_size : int, default=1_000
            The batchsize to generate the datasets
        """
        it_measure = list()

        if Y is not None:
            X, Y = shuffle(X, Y, random_state=self.random_state)
        else:
            X = shuffle(X, random_state=self.random_state)
        for idx in gen_batches(X.shape[0], batch_size, 10):
            if Y is not None:
                it_measure.append(self._fit(X[idx], Y[idx]))
            else:
                it_measure.append(self._fit(X[idx]))

        return np.mean(it_measure)
