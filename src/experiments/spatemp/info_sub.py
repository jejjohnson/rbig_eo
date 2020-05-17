import sys, os
from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(here()))

from typing import Dict, Tuple, Optional, Union, Any
from collections import namedtuple
from sklearn.utils import gen_batches
from joblib import Parallel, delayed

import pathlib
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import joblib
import xarray as xr

# Experiment Functions
from src.data.esdc import get_dataset
from src.features.temporal import select_period, TimePeriod, remove_climatology
from src.features.spatial import select_region, get_spain, get_europe
from sklearn.preprocessing import StandardScaler
from src.models.density import get_rbig_model
from src.models.utils import parallel_predictions
from src.experiments.utils import dict_product, run_parallel_step
from src.features.density import get_density_cubes
from src.features.preprocessing import (
    standardizer_data,
    get_reference_cube,
    get_common_indices,
)
from sklearn.utils import check_random_state

import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=f"%(asctime)s: %(levelname)s: %(message)s",
)

logger = logging.getLogger()
# logger.setLevel(logging.INFO)

SPATEMP = namedtuple("SPATEMP", ["spatial", "temporal", "dimensions"])
RNG = check_random_state(123)
RES_PATH = pathlib.Path(str(root)).joinpath("data/spa_temp/info_earth")


def get_parameters(args) -> Dict:

    parameters = {}
    # ======================
    # Variable
    # ======================
    if args.variable == "gpp":
        parameters["variable"] = ["gross_primary_productivity"]
    elif args.variable == "rm":
        parameters["variable"] = ["root_moisture"]
    elif args.variable == "lst":
        parameters["variable"] = ["land_surface_temperature"]
    elif args.variable == "lai":
        parameters["variable"] = ["leaf_area_index"]
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

    parameters["spatial"] = args.spatial

    parameters["temporal"] = args.temporal

    parameters["subsample"] = args.subsample

    return parameters


def experiment_step(
    params: Dict, smoke_test: bool = False, subsample: Optional[int] = None
) -> Union[Any, Any, Any, Any]:
    # ======================
    # experiment - Data
    # ======================
    # Get DataCube
    logging.info(f"Loading '{params['variable']}' variable")
    datacube = get_dataset(params["variable"])

    # subset datacube (spatially)
    try:
        logging.info(f"Selecting region '{params['region'].name}'")
        datacube = select_region(xr_data=datacube, bbox=params["region"])[
            params["variable"]
        ]
    except:
        logging.info(f"Selecting region 'world'")

    #
    logging.info("Removing climatology...")
    datacube, _ = remove_climatology(datacube)

    # subset datacube (temporally)
    logging.info(f"Selecting temporal period: '{params['period'].name}'")
    datacube = select_period(xr_data=datacube, period=params["period"])

    # get density cubes
    logging.info(
        f"Getting density cubes: S: {params['spatial']}, T: {params['temporal']}"
    )
    if isinstance(datacube, xr.Dataset):
        # print(type(datacube))
        datacube = datacube[params["variable"][0]]

    density_cube_df = get_density_cubes(
        data=datacube, spatial=params["spatial"], temporal=params["temporal"],
    )
    logging.info(f"Total data: {density_cube_df.shape}")

    if smoke_test:
        density_cube_df = density_cube_df.iloc[:1_000]
        logging.info(f"Total data (smoke-test): {density_cube_df.shape}")

    # # standardize data
    logging.info(f"Standardizing data...")
    x_transformer = StandardScaler().fit(density_cube_df.values)

    density_cube_df_norm = pd.DataFrame(
        data=x_transformer.transform(density_cube_df.values),
        columns=density_cube_df.columns.values,
        index=density_cube_df.index,
    )

    # =========================
    # Model - Gaussianization
    # =========================

    # Gaussianize the data
    logging.info(f"Gaussianizing data...")
    t0 = time.time()
    rbig_model = get_rbig_model(
        X=density_cube_df_norm.values, subsample=params["subsample"]
    )
    t1 = time.time() - t0
    logging.info(f"Time Taken: {t1:.2f} secs")

    # get the probability estimates
    logging.info(f"Getting probability estimates...")
    t0 = time.time()
    # add noise

    prob_inputs = density_cube_df_norm.values + 1e-1 * RNG.rand(
        *density_cube_df_norm.values.shape
    )
    logging.info(f"Parallel predictions...")
    X_prob = parallel_predictions(
        X=prob_inputs,
        func=rbig_model.predict_proba,
        batchsize=10_000,
        n_jobs=-1,
        verbose=1,
    )

    t1 = time.time() - t0
    logging.info(f"Time Taken: {t1:.2f} secs")

    X_prob = pd.DataFrame(data=X_prob, index=density_cube_df_norm.index,)

    logging.info(f"Computing Mean...")
    X_prob = X_prob.groupby(level=["lat", "lon"]).mean()
    return rbig_model, x_transformer, X_prob, density_cube_df


def main(args):

    logging.info("Getting parameters...")
    parameters = get_parameters(args)

    logging.info("Getting save path...")
    save_name = (
        f"{args.save}_"
        f"{args.region}_"
        f"{args.variable}_"
        f"{args.period}_"
        f"s{args.subsample}_"
        f"d{args.spatial}{args.spatial}{args.temporal}"
    )

    if args.smoke_test:
        logging.info("Starting smoke test...")
        smoke_test = True
    else:
        smoke_test = False

    rbig_model, x_transformer, X_prob_df, density_cube_df = experiment_step(
        params=parameters, smoke_test=smoke_test, subsample=args.subsample
    )

    # ======================
    # SAVING
    # ======================
    # # Model + Transform
    # logging.info(f"Saving rbig model and transformer...")
    # model = {"rbig": rbig_model, "x_transform": x_transformer, "parameters": parameters}
    # joblib.dump(model, RES_PATH.joinpath(f"models/{save_name}.joblib"))

    # Data
    logging.info(f"Saving data...")
    with open(RES_PATH.joinpath(f"cubes/{save_name}.csv"), "w") as f:
        density_cube_df.to_csv(f, header=True)

    # Probabilities
    logging.info(f"Saving estimated probabilities...")
    with open(RES_PATH.joinpath(f"probs/{save_name}.csv"), "w") as f:
        X_prob_df.to_csv(f, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for GP experiment.")

    parser.add_argument(
        "--res", default="low", type=str, help="Resolution for datacube"
    )

    parser.add_argument(
        "-v", "--variable", default="rm", type=str, help="Variable to use"
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
    parser.add_argument("-sm", "--smoke_test", action="store_true")

    main(parser.parse_args())
