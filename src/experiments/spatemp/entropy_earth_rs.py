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
from src.models.similarity import rbig_h_measures, rbig_h_measures_old
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


RES_PATH = pathlib.Path(str(root)).joinpath("data/spa_temp/info_earth/entropy/rs")


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
    if args.region == "north":
        parameters["region"] = [get_northern_hemisphere()]
    elif args.region == "south":
        parameters["region"] = [get_southern_hemisphere()]
    elif args.region == "world":
        parameters["region"] = ["world"]
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

    # resample
    datacube = datacube.resample(time="1MS").mean()

    # subset datacube (spatially)
    if params["region"] not in ["world"]:
        region_name = params["region"].name
        datacube = select_region(xr_data=datacube, bbox=params["region"])[
            params["variable"]
        ]
    else:
        region_name = "world"

    # remove climatology
    # print(type(datacube))
    datacube, _ = remove_climatology(datacube)
    # print(type(datacube))
    #
    if isinstance(datacube, xr.Dataset):
        # print(type(datacube))
        datacube = datacube[params["variable"]]
        # print(type(datacube))

    # subset datacube (temporally)
    # print(type(datacube))
    # print(datacube)
    datacube = select_period(xr_data=datacube, period=params["period"])

    # get density cubes
    density_cube_df = get_density_cubes(
        data=datacube,
        spatial=params["dimensions"].spatial,
        temporal=params["dimensions"].temporal,
    )

    if smoke_test:
        density_cube_df = density_cube_df.iloc[:10_000]
        logging.info(f"Total data (smoke-test): {density_cube_df.shape}")

    # # standardize data
    x_transformer = StandardScaler().fit(density_cube_df.values)

    density_cube_df_norm = pd.DataFrame(
        data=x_transformer.transform(density_cube_df.values),
        columns=density_cube_df.columns.values,
        index=density_cube_df.index,
    )
    if subsample is not None or smoke_test:
        idx = subset_indices(
            density_cube_df_norm.values, subsample=subsample, random_state=100
        )
        X = density_cube_df_norm.iloc[idx, :].values

    else:
        X = density_cube_df_norm.values

    # =========================
    # Model - Gaussianization
    # =========================

    # Gaussianize the data
    t0 = time.time()
    rbig_h = rbig_h_measures(X, subsample=None, random_state=123, method="new")
    t1 = time.time() - t0

    # Save Results
    results_df = pd.DataFrame(
        {
            "region": region_name,
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

    save_name = (
        f"{args.save}_" + f"{args.region}_" + f"{args.variable}_" + f"{args.period}"
    )
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
                    f"V: {args.variable}, T: {args.period}, "
                    f"R: {args.region}, "
                    f"Spa-Temp: {iparam['dimensions'].temporal}-{iparam['dimensions'].spatial}"
                )

                results_df = experiment_step(
                    params=iparam, smoke_test=False, subsample=args.subsample
                )

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
        "--period", type=str, default="2010", help="Period to do the Gaussianization"
    )
    parser.add_argument("-sm", "--smoke_test", action="store_true")

    main(parser.parse_args())
