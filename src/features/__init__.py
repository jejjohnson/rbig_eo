import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    def get_all(self, ypred, ytest, name: str = "stats"):

        return pd.DataFrame(
            {
                f"{name}_mae": Metrics().get_mae(ytest, ypred),
                f"{name}_mse": Metrics().get_mse(ytest, ypred),
                f"{name}_rmse": Metrics().get_rmse(ytest, ypred),
                f"{name}_r2": Metrics().get_r2(ytest, ypred),
            },
            index=[0],
        )
