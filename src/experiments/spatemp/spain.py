import sys, os
from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(here()))

from typing import Dict, Tuple, Optional
from collections import namedtuple

import pathlib
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

# Experiment Functions
from src.data.esdc import get_dataset
from src.features.temporal import select_period, TimePeriod
from src.features.spatial import select_region, get_spain
from src.models.train_models import get_similarity_scores
from src.experiments.utils import dict_product, run_parallel_step
from src.features.density import get_density_cubes
from src.features.preprocessing import (
    standardizer_data,
    get_reference_cube,
    get_common_indices,
)

SPATEMP = namedtuple("SPATEMP", ["spatial", "temporal", "dimensions"])


RES_PATH = pathlib.Path(str(root)).joinpath("data/spa_temp/trial_experiment")


def get_parameters(variable: str) -> Dict:

    parameters = {}
    if variable == "gpp":
        parameters["variable"] = ["gross_primary_productivity"]
    elif variable == "rm":
        parameters["variable"] = ["root_moisture"]
    elif variable == "sm":
        parameters["variable"] = ["soil_moisture"]
    elif variable == "lst":
        parameters["variable"] = ["land_surface_temperature"]
    elif variable == "precip":
        parameters["variable"] = ["precipitation"]
    elif variable == "wv":
        parameters["variable"] = ["water_vapour"]
    else:
        raise ValueError("Unrecognized variable")

    parameters["region"] = [get_spain()]
    parameters["period"] = [
        TimePeriod(name="201001_201012", start="Jan-2010", end="Dec-2010"),
    ]
    # parameters["spatial"] = [1, 2, 3, 4, 5, 6]
    # parameters["temporal"] = [1, 2, 3, 4, 5, 6]
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
        46,  # 49 dimensions
    ]
    parameters["dimensions"] = [
        SPATEMP(i, j, k)
        for i, j, k in zip(spatial_dimensions, temporal_dimensions, n_dimensions)
    ]
    parameters = list(dict_product(parameters))
    return parameters


def experiment_step(
    params: Dict, smoke_test: bool = False, subsample: Optional[int] = None
) -> None:
    # ======================
    # experiment - Data
    # ======================
    # Get DataCube
    datacube = get_dataset(params["variable"])[params["variable"]]

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
    # # standardize data
    X = density_cube_df.iloc[:, 0][:, np.newaxis]
    Y = density_cube_df.iloc[:, 1:]

    # standardize data
    X, Y = standardizer_data(X=X, Y=Y)

    # ======================
    # experiment - Methods
    # ======================
    res = get_similarity_scores(
        X_ref=X, Y_compare=Y, smoke_test=smoke_test, subsample=subsample
    )

    # Save Results
    results_df = pd.DataFrame(
        {
            "region": params["region"].name,
            "period": params["period"].name,
            "variable": params["variable"],
            "spatial": params["dimensions"].spatial,
            "temporal": params["dimensions"].temporal,
            "n_dimensions": params["dimensions"].dimensions,
            **res,
        },
        index=[0],
    )
    return results_df


def main(args):

    parameters = get_parameters(args.dataset)

    if args.smoke_test:
        iparams = parameters[0]
        result_df = experiment_step(iparams, args.smoke_test)

        # save results

        with open(save_path, "w") as f:
            result_df.to_csv(f, header=True)

    # initialize datast
    else:
        header = True
        mode = "w"
        with tqdm(parameters) as pbar:
            for iparam in pbar:

                save_path = RES_PATH.joinpath(
                    f"{args.save}_{args.dataset}_{iparam['region'].name}.csv"
                )

                pbar.set_description(
                    f"V: {args.dataset}, T: {iparam['period'].name}, "
                    f"R: {iparam['region'].name}, "
                    f"Spa-Temp: {iparam['dimensions'].temporal, iparam['dimensions'].spatial}"
                )

                results_df = experiment_step(
                    params=iparam, smoke_test=False, subsample=args.subsample
                )

                # save results
                with open(save_path, mode) as f:
                    results_df.to_csv(f, header=header)

                header = False
                mode = "a"
                del results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for GP experiment.")

    parser.add_argument(
        "--res", default="low", type=str, help="Resolution for datacube"
    )

    parser.add_argument("-d", "--dataset", default="gpp", type=str, help="Dataset")

    parser.add_argument(
        "-s", "--save", default="exp_v1", type=str, help="Save name for experiment."
    )
    parser.add_argument(
        "--njobs", type=int, default=16, help="number of processes in parallel",
    )
    parser.add_argument(
        "--subsample", type=int, default=10_000, help="subset points to take"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        help="Number of helpful print statements.",
    )
    parser.add_argument("-sm", "--smoke_test", action="store_true")

    main(parser.parse_args())
