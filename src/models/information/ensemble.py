import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import gen_batches, resample
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
from sklearn.utils import shuffle


class Ensemble:
    def __init__(self):
        pass

    def _fit(self, X: np.ndarray) -> BaseEstimator:
        pass

    def _fit_ensemble(self, X: np.ndarray, n_models: int = 10) -> float:
        raise NotImplemented


class Batch:
    """Abstract class to be used to estimate scores in batches.
    
    Parameters
    ----------
    batch_size : int, default = 1_000
        batch size
    
    min_batch_size : int, default = 100
        the minimum batch size to be used for the indices generator

    shuffle : bool, default = True
        option to shuffle the data before doing batches

    random_state : int, default = None
        the random seed when doing the shuffling if option chosen

    summary : str, default = 'mean'
        the way to summarize the scores {'mean', 'median'}
    
    Attributes
    ----------
    raw_scores : np.ndarray
        the raw batchsize scores

    batch_score : float
        the final score after the summary stat (e.g. mean)
    """

    def __init__(
        self,
        batch_size: int = 1_000,
        min_batch_size: int = 100,
        shuffle: bool = True,
        random_state: int = 123,
        summary: str = "mean",
    ):
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.summary = summary

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        IT method to fit to batches. Must be implemented by the user.
        """
        pass

    def _fit_batches(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> float:
        """
        Fits models to inherited class

        Parameters
        ----------
        X : np.ndarray
            The data to be fit.
        
        y : np.ndarray
            The second dataset to be fit

        Returns
        -------
        batch_score : float
            the score after the summary
        """
        it_measure = list()

        # Shuffle dataset
        if self.shuffle:
            if Y is not None:
                X, Y = shuffle(X, Y, random_state=self.random_state)
            else:
                X = shuffle(X, random_state=self.random_state)

        # batch scores
        for idx in gen_batches(X.shape[0], self.batch_size, self.min_batch_size):
            if Y is not None:
                it_measure.append(self._fit(X[idx], Y[idx]))
            else:
                it_measure.append(self._fit(X[idx]))

        # save raw scores
        self.raw_scores = it_measure

        # return summary score
        if self.summary == "mean":
            self.batch_score = np.mean(it_measure)

        elif self.summary == "median":
            self.batch_score = np.median(it_measure)

        else:
            raise ValueError("Unrecognized summarizer: {}".format(self.summary))

        return self.batch_score


class BootStrap:
    def __init__(self, n_iterations=100):
        self.n_iterations = n_iterations

    def _fit(self, X: np.ndarray) -> BaseEstimator:
        pass

    def run_bootstrap(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_size: Optional[int] = 1_000,
    ) -> None:

        raw_scores = list()
        if sample_size is not None:
            n_samples = min(X.shape[0], sample_size)
        else:
            n_samples = X.shape[0]
        for i in range(self.n_iterations):
            if y is None:
                X_sample = resample(X, n_samples=sample_size)
                raw_scores.append(self._fit(X_sample))
            else:
                X_sample, Y_sample = resample(X, y, n_samples=sample_size)
                raw_scores.append(self._fit(X_sample, Y_sample))
        self.raw_scores = raw_scores

        return np.mean(raw_scores)

    def ci(self, p: float) -> Tuple[float, float]:
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1 + p) / 2.0
        l_pval = 1 - u_pval
        l_indx = int(np.floor(self.n_iterations * l_pval))
        u_indx = int(np.floor(self.n_iterations * u_pval))
        return self.raw_scores[l_indx], self.raw_scores[u_indx]

