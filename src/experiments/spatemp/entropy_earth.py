import sys, os
from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(here()))

from typing import Dict, Tuple, Optional, Union, Any
from collections import namedtuple

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
from src.features.temporal import select_period, TimePeriod
from src.features.spatial import (
    select_region,
    get_spain,
    get_europe,
    get_northern_hemisphere,
    get_southern_hemisphere,
)
from sklearn.preprocessing import StandardScaler
from src.features.temporal import remove_climatology
from src.experiments.utils import dict_product, run_parallel_step
from src.features.density import get_density_cubes
from src.features.utils import subset_indices
from src.models.similarity import rbig_h_measures
from src.features.preprocessing import (
    standardizer_data,
    get_reference_cube,
    get_common_indices,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=f"%(asctime)s: %(levelname)s: %(message)s",
)
logger = logging.getLogger()
# logger.setLevel(logging.INFO)

SPATEMP = namedtuple("SPATEMP", ["spatial", "temporal", "dimensions"])


RES_PATH = pathlib.Path(str(root)).joinpath("data/spa_temp/info_earth/entropy")


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
        parameters["region"] = [get_spain()]
    elif args.region == "europe":
        parameters["region"] = [get_europe()]
    elif args.region == "world":
        parameters["region"] = ["world"]
    elif args.region == "north":
        parameters["region"] = [get_northern_hemisphere()]
    elif args.region == "south":
        parameters["region"] = [get_southern_hemisphere()]
    else:
        raise ValueError("Unrecognized region")

    # ======================
    # Period
    # ======================
    if args.period == "2010":

        parameters["period"] = [
            TimePeriod(name="2010", start="Jan-2010", end="Dec-2010")
        ]

    elif args.period == "2002_2010":
        parameters["period"] = [
            TimePeriod(name="2002_2010", start="Jan-2002", end="Dec-2010")
        ]
    if args.resample:
        spatial_dimensions = [
            1,  # 1 dimension
            2,
            1,  # 4 Dimensions
            3,
            2,
            2,
            1,  # 9 Dimensions
            4,
            3,
            2,
            1,  # 16 total dimensions
        ]
        temporal_dimensions = [
            1,  # 1 dimension
            1,
            4,  # 4 dimensions
            1,
            2,
            3,
            9,  # 9 dimensions
            1,
            2,
            4,
            12,  # 16 dimensions
        ]
        n_dimensions = [
            1,
            4,
            4,  # 4 dimensions
            9,
            9,
            9,
            9,  # 9 dimensions
            16,
            16,
            16,
            16,  # 16 dimensions
        ]
    else:
        spatial_dimensions = [
            1,
            2,
            1,  # 4 Dimensions
            3,
            2,
            1,  # 9 Dimensions
            4,
            3,
            2,
            1,  # 16 total dimensions
            5,
            3,
            2,
            1,  # 25 total dimensions
            6,
            4,
            3,
            2,
            1,  # 36 total dimensions
            7,
            5,
            4,
            3,
            2,
            1,  # 49 total dimensions
        ]
        temporal_dimensions = [
            1,
            1,
            4,  # 4 dimensions
            1,
            2,
            9,  # 9 dimensions
            1,
            2,
            4,
            16,  # 16 dimensions
            1,
            3,
            6,
            25,  # 25 dimensions
            1,
            2,
            4,
            9,
            36,  # 36 dimensions
            1,
            2,
            3,
            5,
            12,
            46,  # 49 dimensions
        ]
        n_dimensions = [
            1,
            4,
            4,  # 4 dimensions
            9,
            9,
            9,  # 9 dimensions
            16,
            16,
            16,
            16,  # 16 dimensions
            25,
            25,
            25,
            25,  # 25 dimensions
            36,
            36,
            36,
            36,
            36,  # 36 dimensions
            49,
            49,
            49,
            49,
            49,
            49,  # 49 dimensions
        ]
    parameters["dimensions"] = [
        SPATEMP(i, j, k)
        for i, j, k in zip(spatial_dimensions, temporal_dimensions, n_dimensions)
    ]
    parameters = list(dict_product(parameters))
    return parameters


def experiment_step(parameters: Dict, args: argparse.Namespace,) -> pd.DataFrame:

    # ======================
    # experiment - Data
    # ======================
    # Get DataCube
    datacube = get_dataset([parameters["variable"]])

    # ======================
    # RESAMPLE
    # ======================
    if args.resample:
        datacube = datacube.resample(time=args.resample).mean()

    # ======================
    # SPATIAL SUBSET
    # ======================
    if parameters["region"] not in ["world"]:
        region_name = parameters["region"].name
        datacube = select_region(xr_data=datacube, bbox=parameters["region"])[
            parameters["variable"]
        ]
    else:
        region_name = "world"

    # ======================
    # CLIMATOLOGY (TEMPORAL)
    # ======================
    if args.remove_climatology:
        datacube, _ = remove_climatology(datacube)
    # print(type(datacube))
    #
    # ======================
    # TEMPORAL SUBSET
    # ======================
    datacube = select_period(xr_data=datacube, period=parameters["period"])

    # ======================
    # DENSITY CUBES
    # ======================
    if isinstance(datacube, xr.Dataset):
        # print(type(datacube))
        datacube = datacube[parameters["variable"]]

    density_cube_df = get_density_cubes(
        data=datacube,
        spatial=parameters["dimensions"].spatial,
        temporal=parameters["dimensions"].temporal,
    )

    # ======================
    # STANDARDIZE DATA
    # ======================
    x_transformer = StandardScaler().fit(density_cube_df.values)

    density_cube_df_norm = pd.DataFrame(
        data=x_transformer.transform(density_cube_df.values),
        columns=density_cube_df.columns.values,
        index=density_cube_df.index,
    )
    # ======================
    # SUBSAMPLE DATA
    # ======================
    if args.subsample is not None:
        idx = subset_indices(
            density_cube_df_norm.values, subsample=args.subsample, random_state=100
        )
        if idx is not None:
            X = density_cube_df_norm.iloc[idx, :].values
        else:
            X = density_cube_df_norm.values
    else:
        X = density_cube_df_norm.values

    # =========================
    # Model - Gaussianization
    # =========================
    # Gaussianize the data
    t0 = time.time()
    rbig_h = rbig_h_measures(X, random_state=123, method=args.method)
    t1 = time.time() - t0

    # Save Results
    results_df = pd.DataFrame(
        {
            "region": region_name,
            "period": parameters["period"].name,
            "variable": parameters["variable"],
            "spatial": parameters["dimensions"].spatial,
            "temporal": parameters["dimensions"].temporal,
            "n_dimensions": parameters["dimensions"].dimensions,
            "n_samples": X.shape[0],
            "entropy": rbig_h,
            "time": t1,
        },
        index=[0],
    )
    return results_df


def main(args):

    parameters = get_parameters(args)

    save_name = (
        f"{args.save}_" + f"{args.region}_" + f"{args.variable}_" + f"{args.period}"
    )
    if args.subsample:
        save_name += f"_s{int(args.subsample / 1_000)}k"
    if args.resample:
        save_name += f"_rs{args.resample}"
    if args.remove_climatology:
        save_name += f"_rc"

    header = True
    mode = "w"
    if args.smoke_test:
        # print(parameters)
        iparam = parameters[0]
        print(iparam)
        result_df = experiment_step(parameters=iparam, args=args)
        with open(RES_PATH.joinpath(f"sm_{save_name}.csv"), mode) as f:
            result_df.to_csv(f, header=header)
    else:

        with tqdm(parameters) as pbar:
            for iparam in pbar:

                pbar.set_description(
                    f"V: {args.variable}, T: {args.period}, "
                    f"R: {args.region}, "
                    f"Spa-Temp: {iparam['dimensions'].temporal}-{iparam['dimensions'].spatial}"
                )

                results_df = experiment_step(parameters=iparam, args=args)

                # save results
                with open(RES_PATH.joinpath(f"{save_name}.csv"), mode) as f:
                    results_df.to_csv(f, header=header)

                header = False
                mode = "a"
                del results_df


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
        "--subsample", type=int, default=None, help="subset points to take"
    )
    parser.add_argument(
        "--region", type=str, default="spain", help="Region to be Gaussianized"
    )
    parser.add_argument(
        "--period", type=str, default="2010", help="Period to do the Gaussianization"
    )
    parser.add_argument(
        "-rs", "--resample", type=str, default=None, help="Resample Frequency"
    )
    parser.add_argument("-m", "--method", type=str, default="old", help="RBIG Method")
    parser.add_argument("-sm", "--smoke-test", action="store_true")
    parser.add_argument("-tm", "--temporal-mean", action="store_true")
    parser.add_argument("-rc", "--remove-climatology", action="store_true")

    main(parser.parse_args())
