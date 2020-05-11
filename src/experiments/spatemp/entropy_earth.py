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

# Experiment Functions
from src.data.esdc import get_dataset
from src.features.temporal import select_period, TimePeriod
from src.features.spatial import select_region, get_spain, get_europe
from sklearn.preprocessing import StandardScaler
from src.models.density import get_rbig_model
from src.experiments.utils import dict_product, run_parallel_step
from src.features.density import get_density_cubes
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
    elif args.variable == "sm":
        parameters["variable"] = ["soil_moisture"]
    elif args.variable == "lst":
        parameters["variable"] = ["land_surface_temperature"]
    elif args.variable == "precip":
        parameters["variable"] = ["precipitation"]
    elif args.variable == "wv":
        parameters["variable"] = ["water_vapour"]
    else:
        raise ValueError("Unrecognized variable")

    # ======================
    # Region
    # ======================
    if args.region == "spain":
        parameters["region"] = [get_spain()]
    elif args.region == "europe":
        parameters["region"] = [get_europe()]
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

    spatial_dimensions = [
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


def experiment_step(
    params: Dict, smoke_test: bool = False, subsample: Optional[int] = None
) -> pd.DataFrame:
    # ======================
    # experiment - Data
    # ======================
    # Get DataCube
    datacube = get_dataset([params["variable"]])

    # subset datacube (spatially)
    datacube = select_region(xr_data=datacube, bbox=params["region"])[
        params["variable"]
    ]

    # subset datacube (temporally)
    datacube = select_period(xr_data=datacube, period=params["period"]).compute()

    # get density cubes
    density_cube_df = get_density_cubes(
        data=datacube,
        spatial=params["dimensions"].spatial,
        temporal=params["dimensions"].temporal,
    )

    if smoke_test:
        density_cube_df = density_cube_df.iloc[:1_000]
        logging.info(f"Total data (smoke-test): {density_cube_df.shape}")

    # # standardize data
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
    t0 = time.time()
    rbig_h = rbig_h_measures(
        density_cube_df_norm.values, subsample=subsample, random_state=123
    )
    t1 = time.time() - t0

    # Save Results
    results_df = pd.DataFrame(
        {
            "region": params["region"].name,
            "period": params["period"].name,
            "variable": params["variable"],
            "spatial": params["dimensions"].spatial,
            "temporal": params["dimensions"].temporal,
            "n_dimensions": params["dimensions"].dimensions,
            "n_samples": density_cube_df_norm.shape[0],
            "entropy": rbig_h,
            "time": t1,
        },
        index=[0],
    )
    return results_df


def main(args):

    parameters = get_parameters(args)

    save_name = f"{args.save}_" f"{args.region}_" f"{args.variable}_" f"{args.period}"
    header = True
    mode = "w"
    if args.smoke_test:

        iparam = parameters[0]
        result_df = experiment_step(
            params=iparam, smoke_test=True, subsample=args.subsample
        )
        with open(RES_PATH.joinpath(f"entropy/{save_name}.csv"), mode) as f:
            result_df.to_csv(f, header=header)
    else:

        with tqdm(parameters) as pbar:
            for iparam in pbar:

                pbar.set_description(
                    f"V: {args.variable}, T: {iparam['period'].name}, "
                    f"R: {iparam['region'].name}, "
                    f"Spa-Temp: {iparam['dimensions'].temporal}-{iparam['dimensions'].spatial}"
                )

                results_df = experiment_step(
                    params=iparam, smoke_test=False, subsample=args.subsample
                )

                # save results
                with open(RES_PATH.joinpath(f"entropy/{save_name}.csv"), mode) as f:
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
        "-v", "--variable", default="rm", type=str, help="Variable to use"
    )

    parser.add_argument(
        "-s", "--save", default="v0", type=str, help="Save name for experiment."
    )
    parser.add_argument(
        "--njobs", type=int, default=-1, help="number of processes in parallel",
    )
    parser.add_argument(
        "--subsample", type=int, default=10_000, help="subset points to take"
    )
    parser.add_argument(
        "--region", type=str, default="spain", help="Region to be Gaussianized"
    )
    parser.add_argument(
        "--period", type=str, default="2010", help="Period to do the Gaussianization"
    )
    parser.add_argument("-sm", "--smoke_test", action="store_true")

    main(parser.parse_args())
