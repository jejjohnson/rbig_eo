import sys, os
from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(here()))

from typing import Dict, Tuple

import pathlib
import argparse
import pandas as pd
from tqdm import tqdm

# Experiment Functions
from src.data.esdc import get_dataset
from src.features.temporal import (
    select_period,
    get_smoke_test_time,
    get_fall_time,
    get_spring_time,
    get_winter_time,
    get_summer_time,
)
from src.features.spatial import select_region, get_europe
from src.models.train_models import get_similarity_scores
from src.experiments.utils import dict_product, run_parallel_step
from src.features.density import get_density_cubes
from src.features.preprocessing import (
    standardizer_data,
    get_reference_cube,
    get_common_indices,
)


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
    else:
        raise ValueError("Unrecognized variable")

    parameters["region"] = [get_europe()]
    parameters["period"] = [
        get_winter_time(),
        get_spring_time(),
        get_summer_time(),
        get_fall_time(),
    ]
    parameters["spatial"] = [1, 2, 3, 4, 5, 6]
    parameters["temporal"] = [1, 2, 3, 4, 5, 6]

    parameters = list(dict_product(parameters))
    return parameters


def experiment_step(params: Dict, smoke_test: bool = False) -> None:
    # ======================
    # experiment - Data
    # ======================
    # Get DataCube
    datacube = get_dataset(params["variable"])

    # subset datacube (spatially)
    datacube = select_region(xr_data=datacube, bbox=params["region"])[
        params["variable"]
    ]

    # subset datacube (temporally)
    datacube = select_period(xr_data=datacube, period=params["period"])

    # get datacubes
    reference_cube_df = get_reference_cube(data=datacube)

    # get density cubes
    density_cube_df = get_density_cubes(
        data=datacube, spatial=params["spatial"], temporal=params["temporal"]
    )

    # get reference dataframe
    X, Y = get_common_indices(
        reference_df=reference_cube_df, density_df=density_cube_df
    )

    # standardize data
    X, Y = standardizer_data(X=X, Y=Y)

    # ======================
    # experiment - Methods
    # ======================
    res = get_similarity_scores(X_ref=X, Y_compare=Y, smoke_test=smoke_test)

    # Save Results
    results_df = pd.DataFrame(
        {
            "region": params["region"].name,
            "period": params["period"].name,
            "variable": params["variable"],
            "spatial": params["spatial"],
            "temporal": params["temporal"],
            **res,
        },
        index=[0],
    )
    return results_df


def main(args):

    save_path = RES_PATH.joinpath(f"{args.save}_{args.dataset}_europe.csv")

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
                pbar.set_description(
                    f"V: {args.dataset}, T: {iparam['period'].name}, R: {iparam['region'].name}, Spa-Temp: {iparam['spatial'], iparam['temporal']}"
                )

                results_df = experiment_step(params=iparam, smoke_test=False)

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
        "-v",
        "--verbose",
        type=int,
        default=1,
        help="Number of helpful print statements.",
    )
    parser.add_argument("-sm", "--smoke_test", action="store_true")

    main(parser.parse_args())
