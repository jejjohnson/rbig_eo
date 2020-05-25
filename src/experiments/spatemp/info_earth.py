import sys, os
from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(here()))

from typing import Dict, Optional, Union, Any
from collections import namedtuple
import pathlib
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import joblib
import xarray as xr
import logging

# Experiment Functions
from src.data.esdc import get_dataset
from src.features.temporal import select_period, TimePeriod, remove_climatology
from src.features.spatial import (
    get_spain,
    select_region,
    get_europe,
    get_northern_hemisphere,
    get_southern_hemisphere,
)
from sklearn.preprocessing import StandardScaler
from src.models.density import get_rbig_model
from src.models.utils import parallel_predictions
from src.features.utils import subset_indices
from src.features.density import get_density_cubes, get_information_cubes
from src.features.preprocessing import (
    standardizer_data,
    get_reference_cube,
    get_common_indices,
)
from sklearn.utils import check_random_state


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=f"%(asctime)s: %(levelname)s: %(message)s",
)

logger = logging.getLogger()
# logger.setLevel(logging.INFO)

SPATEMP = namedtuple("SPATEMP", ["spatial", "temporal", "dimensions"])
RNG = check_random_state(123)
RES_PATH = pathlib.Path(str(root)).joinpath("data/spa_temp/info_earth/world")


def get_parameters(args) -> Dict:

    parameters = {}

    # ======================
    # Variable
    # ======================
    if args.variable == "gpp":

        parameters["variable"] = ["gross_primary_productivity"]

    elif args.variable == "sm":

        parameters["variable"] = ["soil_moisture"]

    elif args.variable == "lst":

        parameters["variable"] = ["land_surface_temperature"]

    elif args.variable == "lai":

        parameters["variable"] = ["leaf_area_index"]

    elif args.variable == "rm":

        parameters["variable"] = ["root_moisture"]

    elif args.variable == "precip":

        parameters["variable"] = ["precipitation"]

    else:
        raise ValueError("Unrecognized variable")

    # ======================
    # Region
    # ======================
    if args.region == "spain":
        parameters["region"] = get_spain()
    elif args.region == "europe":
        parameters["region"] = get_europe()
    elif args.region == "world":
        parameters["region"] = ["world"]
    elif args.region == "north":
        parameters["region"] = get_northern_hemisphere()
    elif args.region == "south":
        parameters["region"] = get_southern_hemisphere()
    else:
        raise ValueError("Unrecognized region")

    # ======================
    # Period
    # ======================
    if args.period == "2010":

        parameters["period"] = TimePeriod(name="2010", start="Jan-2010", end="Dec-2010")

    elif args.period == "2002_2010":
        parameters["period"] = TimePeriod(
            name="2002_2010", start="Jan-2002", end="Dec-2010"
        )

    return parameters


def experiment_step(args: argparse.Namespace,) -> Union[Any, Any]:

    logging.info(f"Extracting Parameters")
    parameters = get_parameters(args)
    # ======================
    # experiment - Data
    # ======================
    # Get DataCube
    logging.info(f"Loading '{parameters['variable'][0]}' variable")
    datacube = get_dataset(parameters["variable"])

    # ======================
    # RESAMPLE
    # ======================
    if args.resample:
        logging.info(f"Resampling datacube...")
        datacube = datacube.resample(time="1MS").mean()

    # ======================
    # SPATIAL SUBSET
    # ======================
    try:
        logging.info(f"Selecting region '{parameters['region'].name}'")
        datacube = select_region(xr_data=datacube, bbox=parameters["region"])[
            parameters["variable"]
        ]
    except:
        logging.info(f"Selecting region 'world'")

    # ======================
    # CLIMATOLOGY (TEMPORAL)
    # ======================
    if args.clima:
        logging.info("Removing climatology...")
        datacube, _ = remove_climatology(datacube)

    # ======================
    # TEMPORAL SUBSET
    # ======================
    logging.info(f"Selecting temporal period: '{parameters['period'].name}'")
    datacube = select_period(xr_data=datacube, period=parameters["period"])

    # ======================
    # DENSITY CUBES
    # ======================
    logging.info(f"Getting density cubes: S: {args.spatial}, T: {args.temporal}")
    if isinstance(datacube, xr.Dataset):
        datacube = datacube[parameters["variable"][0]]

    density_cube_df = get_density_cubes(
        data=datacube, spatial=args.spatial, temporal=args.temporal,
    )
    logging.info(f"Total data: {density_cube_df.shape}")

    # ======================
    # STANDARDIZE DATA
    # ======================
    logging.info(f"Standardizing data...")
    x_transformer = StandardScaler().fit(density_cube_df.values)

    density_cube_df_norm = pd.DataFrame(
        data=x_transformer.transform(density_cube_df.values),
        columns=density_cube_df.columns.values,
        index=density_cube_df.index,
    )

    # ======================
    # SUBSAMPLE DATA
    # ======================
    if args.smoke_test:
        logging.info(f"Smoke Test...")
        logging.info(f"Subsampling datacube...")
        idx = subset_indices(
            density_cube_df_norm.values, subsample=1000, random_state=100
        )
        X = density_cube_df_norm.iloc[idx, :].values
        index = density_cube_df_norm.iloc[idx, :].index
    elif args.subsample is not None:
        logging.info(f"Subsampling datacube...")
        idx = subset_indices(
            density_cube_df_norm.values, subsample=args.subsample, random_state=100
        )
        X = density_cube_df_norm.iloc[idx, :].values
        index = density_cube_df_norm.index
    else:
        X = density_cube_df_norm.values
        index = density_cube_df_norm.index

    logging.info(f"Input shape: {X.shape}")
    parameters["input_shape"] = X.shape

    # =========================
    # Model - Gaussianization
    # =========================
    # Gaussianize the data
    logging.info(f"Gaussianizing data...")
    t0 = time.time()
    rbig_model = get_rbig_model(X=X, method=args.method)

    rbig_model.fit(X)
    t1 = time.time() - t0
    logging.info(f"Time Taken: {t1:.2f} secs")
    parameters["rbig_fit_time"] = t1

    # =========================
    # PROB ESTIMATES
    # =========================
    logging.info(f"Getting probability estimates...")
    t0 = time.time()
    # add noise
    if args.add_noise:
        logging.info(f"Adding noise to values for probability...")
        density_cube_df_norm.values += 1e-1 * RNG.rand(
            *density_cube_df_norm.values.shape
        )
    logging.info(f"Parallel predictions...")
    if args.smoke_test:
        X_prob = parallel_predictions(
            X=X, func=rbig_model.predict_proba, batchsize=100, n_jobs=-1, verbose=1,
        )
    else:
        X_prob = parallel_predictions(
            X=density_cube_df_norm.values,
            func=rbig_model.predict_proba,
            batchsize=10_000,
            n_jobs=-1,
            verbose=1,
        )

    t1 = time.time() - t0
    logging.info(f"Time Taken: {t1:.2f} secs")
    parameters["prob_size"] = density_cube_df_norm.values.shape
    parameters["rbig_predict_time"] = t1

    X_prob = pd.DataFrame(data=X_prob, index=index, columns=["probability"])

    # returning density cubes
    logging.info(f"Getting information cubes.")
    X_prob = get_information_cubes(X_prob, time=args.temporal_mean)

    X_prob.attrs = parameters
    return X_prob


def main(args):

    logging.info("Getting parameters...")

    logging.info("Getting save path...")
    save_name = (
        f"{args.save}_"
        f"{args.region}_"
        f"{args.variable}_"
        f"{args.period}_"
        f"s{args.subsample}_"
        f"d{args.spatial}{args.spatial}{args.temporal}"
    )
    if args.resample:
        save_name += f"_rs{args.resample}"

    X_prob = experiment_step(args=args)

    # ======================
    # SAVING
    # ======================
    # # Model + Transform
    # logging.info(f"Saving rbig model and transformer...")
    # model = {"rbig": rbig_model, "x_transform": x_transformer, "parameters": parameters}
    # joblib.dump(model, RES_PATH.joinpath(f"models/{save_name}.joblib"))

    # # Data
    # logging.info(f"Saving data...")
    # with open(RES_PATH.joinpath(f"cubes/{save_name}.csv"), "w") as f:
    #     density_cube_df.to_csv(f, header=True)

    # Probabilities
    logging.info(f"Saving estimated probabilities...")
    # with open(RES_PATH.joinpath(f"probs/{save_name}.h5"), "w") as f:
    X_prob.to_netcdf(RES_PATH.joinpath(f"prob_cubes/{save_name}.h5"), "w")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for GP experiment.")

    parser.add_argument(
        "--res", default="low", type=str, help="Resolution for datacube"
    )

    parser.add_argument(
        "-v", "--variable", default="gpp", type=str, help="Variable to use"
    )

    parser.add_argument(
        "-s", "--save", default="v0", type=str, help="Save name for experiment."
    )
    parser.add_argument(
        "--njobs", type=int, default=-1, help="number of processes in parallel",
    )
    parser.add_argument(
        "--subsample", type=int, default=200_000, help="subset points to take"
    )
    parser.add_argument(
        "--region", type=str, default="spain", help="Region to be Gaussianized"
    )
    parser.add_argument(
        "--temporal", type=int, default=1, help="Number of temporal dimensions",
    )
    parser.add_argument(
        "--spatial", type=int, default=1, help="Number of spatial dimensions"
    )
    parser.add_argument(
        "--period", type=str, default="2010", help="Period to do the Gaussianization"
    )
    parser.add_argument(
        "--hemisphere", type=str, default="top", help="Hemisphere for data"
    )
    parser.add_argument(
        "-rs", "--resample", type=str, default=None, help="Resample Frequency"
    )
    parser.add_argument("-m", "--method", type=str, default="old", help="RBIG Method")
    parser.add_argument("-sm", "--smoke-test", action="store_true")
    parser.add_argument("-tm", "--temporal-mean", action="store_true")
    parser.add_argument("-c", "--clima", action="store_true")
    parser.add_argument("-n", "--add-noise", action="store_true")

    main(parser.parse_args())
