import sys, os

cwd = os.getcwd()
sys.path.insert(0, f"{cwd}/../../")
sys.path.insert(0, "/home/emmanuel/code/py_esdc")
sys.path.insert(0, "/home/emmanuel/code/gp_model_zoo")

import argparse

# standard python packages
import xarray as xr
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#
from src.models.spatemp.train_models import Metrics, train_sparse_gp, batch_predict
from src.models.train_models import run_rbig_models
from src.features.spatemp.build_features import (
    select_region,
    normalize_inputs,
    normalize_outputs,
)

# esdc tools
from esdc.subset import select_pixel
from esdc.shape import ShapeFileExtract, rasterize
from esdc.transform import DensityCubes
from tqdm import tqdm


HIGH_RES_CUBE = "/media/disk/databases/ESDC/esdc-8d-0.083deg-1x2160x4320-2.0.0.zarr"
LOW_RES_CUBE = "/media/disk/databases/ESDC/esdc-8d-0.25deg-1x720x1440-2.0.0.zarr"
SAVE_PATH = "/home/emmanuel/projects/2020_rbig_rs/data/spa_temp/"


def main(args):

    # Get DataCubes
    if args.res == "high":
        datacube = xr.open_zarr(HIGH_RES_CUBE)
    elif args.res == "low":
        datacube = xr.open_zarr(LOW_RES_CUBE)
    else:
        raise ValueError("Unrecognized res: ", args.res)

    # Experimental parameters
    trials = np.linspace(1, args.trials, args.trials, dtype=int)
    resolutions = [(7, 1), (5, 2), (4, 3), (3, 5), (2, 11), (1, 46)]

    # select variables
    columns = [
        "land_surface_temperature",
        "gross_primary_productivity",
        "precipitation",
        "soil_moisture",
    ]
    datacube = datacube[columns]

    results_df = pd.DataFrame()

    # select region
    regions = ["europe"]

    for iregion in regions:
        datacube = select_region(datacube, region=regions[0])
        # print(datacube)
        # Loop through variables
        with tqdm(datacube.data_vars.items()) as vars_bar:
            for (iname, icube) in vars_bar:

                # get density cubes

                for (ispatial, itemporal) in resolutions:

                    # initialize minicuber
                    minicuber = DensityCubes(
                        spatial_window=ispatial, time_window=itemporal
                    )

                    mini_cubes = minicuber.get_minicubes(icube)

                    # get data and prediction points
                    y = mini_cubes.iloc[:, 0][:, np.newaxis]
                    X = mini_cubes.iloc[:, 1:]
                    # print(X.shape, y.shape)
                    for itrial in trials:

                        # get minicube
                        xtrain, xtest, ytrain, ytest = train_test_split(
                            X,
                            y,
                            train_size=args.train,
                            test_size=args.test,
                            random_state=itrial,
                        )

                        # Normalize Data
                        xtrain, xtest = normalize_inputs(xtrain, xtest)
                        ytrain, ytest = normalize_outputs(ytrain, ytest)

                        # print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
                        # break
                        # train and test
                        ypred = train_sparse_gp(
                            xtrain,
                            ytrain,
                            xtest,
                            n_inducing=args.inducing,
                            restarts=args.restarts,
                        )

                        # print(ypred.shape)

                        # get metrics
                        stats = Metrics().get_all(ypred.squeeze(), ytest.squeeze())

                        # Entropy
                        tc, h, _ = run_rbig_models(
                            xtrain, measure="t", random_state=123
                        )

                        r2 = stats["r2"].values[0]
                        # append and save results
                        results_df = results_df.append(
                            {
                                "region": iregion,
                                "dimensions": xtrain.shape[1],
                                "variable": iname,
                                "spatial": ispatial,
                                "temporal": itemporal,
                                "train_size": xtrain.shape[0],
                                "seed": args.seed,
                                "inducing": args.inducing,
                                "test_size": xtest.shape[0],
                                "r2": stats["r2"].values[0],
                                "mse": stats["mse"].values[0],
                                "mae": stats["mae"].values[0],
                                "rmse": stats["rmse"].values[0],
                                "trial": itrial,
                                "h": h,
                                "tc": tc,
                            },
                            ignore_index=True,
                        )

                        results_df.to_csv(SAVE_PATH + args.save + ".csv")
                        postfix = dict(
                            Region=iregion,
                            Variable=iname,
                            SpaTemp=f"{ispatial}-{itemporal}",
                            Trial=itrial,
                            H=f"{h:.3f}",
                            TC=f"{tc:.3f}",
                            R2=f"{r2:.2f}",
                        )
                        vars_bar.set_postfix(postfix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for GP experiment.")

    parser.add_argument(
        "--res", default="low", type=str, help="Resolution for datacube"
    )

    parser.add_argument(
        "--train", default=10_000, type=int, help="Number of training points for GP"
    )
    parser.add_argument(
        "--test", default=100_000, type=int, help="Number of testing points for GP"
    )
    parser.add_argument(
        "--seed", default=123, type=int, help="Random state for reproducibility"
    )

    parser.add_argument(
        "--restarts", default=0, type=int, help="Number of restarts for GP optimizer"
    )
    parser.add_argument(
        "--inducing",
        default=300,
        type=int,
        help="Number of inducing points for GP algorithm",
    )

    parser.add_argument(
        "--trials",
        default=1,
        type=int,
        help="Number of trails for selecting training points",
    )

    parser.add_argument(
        "--save", default="exp_v1", type=str, help="Save name for experiment."
    )

    main(parser.parse_args())

