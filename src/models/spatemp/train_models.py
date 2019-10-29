from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd


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
