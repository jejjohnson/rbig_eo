import pytest
import numpy as np
from src.models.utils import parallel_predictions


@pytest.mark.utils
def test_parallel():
    X = np.random.randn(1_000, 10)

    def f(x):
        return x

    X_ = parallel_predictions(X, func=f, n_jobs=2, batchsize=10)

    np.testing.assert_array_equal(X, X_)
