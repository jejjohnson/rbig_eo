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
    batch_scores : np.ndarray
        the raw batchsize scores

    score : float
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
        score : float
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
        self.batch_scores = np.ndarray(it_measure)

        # return summary score
        if self.summary == "mean":
            self.score = np.mean(it_measure)

        elif self.summary == "median":
            self.score = np.median(it_measure)

        else:
            raise ValueError("Unrecognized summarizer: {}".format(self.summary))

        return self.score
