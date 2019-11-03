import sys, os

cwd = os.getcwd()
sys.path.insert(0, "/home/emmanuel/code/gp_model_zoo")

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from gpy.sparse import SparseGPR
import GPy


def train_sparse_gp(
    X: np.ndarray,
    y: np.ndarray,
    Xtest: np.ndarray,
    n_inducing: int = 300,
    restarts: int = 10,
) -> np.ndarray:

    n_dims = X.shape[1]
    kernel = GPy.kern.RBF(input_dim=n_dims, ARD=False)
    inference = "vfe"
    n_inducing = 300
    verbose = 0
    max_iters = 5_000

    # initialize GP Model
    sgp_model = SparseGPR(
        kernel=kernel,
        inference=inference,
        n_inducing=n_inducing,
        verbose=verbose,
        max_iters=max_iters,
        n_restarts=restarts,
    )

    # train GP model
    sgp_model.fit(X, y)

    # make predictions
    return batch_predict(sgp_model, Xtest, 10_000)


def batch_predict(model, Xs, batch_size=10_000):
    ms = []
    n = max(len(Xs) / batch_size, 1)  # predict in small batches
    # with tqdm() as bar:
    for xs in np.array_split(Xs, n):
        m = model.predict(xs)
        ms.append(m)

    return np.vstack(ms)


class Metrics:
    @staticmethod
    def get_r2(ypred, ytest):
        return r2_score(ytest, ypred)

    @staticmethod
    def get_mae(ypred, ytest):
        return mean_absolute_error(ytest, ypred)

    @staticmethod
    def get_mse(ypred, ytest):
        return mean_squared_error(ytest, ypred)

    @staticmethod
    def get_rmse(ypred, ytest):
        return np.sqrt(mean_squared_error(ytest, ypred))

    def get_all(self, ypred, ytest):

        return pd.DataFrame(
            {
                "mae": Metrics().get_mae(ypred, ytest),
                "mse": Metrics().get_mse(ypred, ytest),
                "rmse": Metrics().get_rmse(ypred, ytest),
                "r2": Metrics().get_r2(ypred, ytest),
            },
            index=[0],
        )
