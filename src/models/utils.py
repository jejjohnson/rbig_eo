from typing import Tuple, Optional, Callable
from sklearn.utils import check_random_state
import numpy as np
from sklearn.utils import gen_batches, check_array
from joblib import Parallel, delayed


def subset_indices(
    X: np.ndarray, subsample: Optional[int] = None, random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:

    if subsample is not None and subsample < X.shape[0]:
        rng = check_random_state(random_state)
        indices = np.arange(X.shape[0])
        subset_indices = rng.permutation(indices)[:subsample]
        X = X[subset_indices, :]

    return X


def parallel_predictions(
    X: np.ndarray, func: Callable, batchsize: int = 10_000, n_jobs: int = 1, verbose=0
) -> np.ndarray:
    """Function to do parallel predictions
    Primary use was for predictions but any function will do with
    one inputs.
    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        input data to be predicted
    func : Callable
        the callable function
    batchsize : int, default=10_000
        the size of the batches
    n_jobs : int, default=1
        the number of jobs
    verbose : int, default=0
        the verbosity of the parallel predictions
    """
    X = check_array(X, ensure_2d=True)
    # get indices slices
    slices = list(gen_batches(X.shape[0], batchsize))

    # delayed jobs function for predictions
    jobs = (delayed(func)(X[islice, :]) for islice in slices)

    # parallel function
    parallel = Parallel(verbose=verbose, n_jobs=n_jobs)

    # do parallel predictions
    results = parallel(jobs)

    # return as array of inputs
    # print(len(results))
    # print(results[0].shape)
    results = np.concatenate(results, 0).reshape(-1, 1)
    # print(results.shape)
    msg = f"Sizes don't match: {results.shape}=/={X.shape}"
    assert results.shape[0] == X.shape[0], msg
    return results


if __name__ == "__main__":
    X = np.random.randn(1_000, 1)

    def f(x):
        return x

    X_ = parallel_predictions(X, func=f, n_jobs=2, batchsize=100)

    np.testing.assert_array_almost_equal(X, X_)
