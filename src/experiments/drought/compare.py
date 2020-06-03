import sys, os
from pyprojroot import here
import logging
import pathlib

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

PATH = pathlib.Path(str(here()))
# root = here(project_files=[".here"])
sys.path.append(str(here()))


import argparse
import numpy as np

# drought tools
from src.data.drought.loader import DataLoader
from src.features.drought.build_features import (
    get_cali_geometry,
    mask_datacube,
    remove_climatology,
    get_density_cubes,
    get_common_elements_many,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.train_models import get_similarity_scores
from src.models.similarity import univariate_stats
from tqdm import tqdm
from scipy import stats
from src.experiments.utils import dict_product
import itertools

RES_PATH = PATH.joinpath("data/drought/results/")


def main(args):

    # get save name
    SAVE_NAME = RES_PATH.joinpath(
        args.save + f"_t{args.temporal}_s{args.spatial}_c{args.compare}.csv"
    )

    # Load data
    logger.info("Loading datacube...")
    drought_cube = DataLoader().load_data(args.region, args.sampling)

    # get cali geometry
    logger.info("Getting shapefile...")
    if args.region in ["conus"]:
        shape_file = get_cali_geometry()
    else:
        raise ValueError("Unrecognized region.")

    # subset datacube with cali
    logger.info(f"Masking dataset with {args.region} shapefile.")
    drought_cube = mask_datacube(drought_cube, shape_file)

    # do interpolation
    logger.info(f"Interpolating time dims with {args.interp_method} method")
    drought_cube = drought_cube.interpolate_na(dim="time", method=args.interp_method)

    # Remove climatology
    logger.info(f"Removing climatology")
    drought_cube, _ = remove_climatology(drought_cube)

    # drought_years
    drought_years = {
        "2010": False,
        "2011": False,
        "2012": True,
        "2013": False,
        "2014": True,
        "2015": True,
    }
    # # MI elements
    # variables_names = ["VOD", "NDVI", "LST", "SM"]

    # ========================
    # Experimental Parameters
    # ========================
    parameters = {}
    parameters["cubes"] = list(drought_cube.groupby("time.year"))
    parameters["temporal"] = np.arange(1, args.temporal + 1)
    parameters["spatial"] = np.arange(1, args.spatial + 1)

    parameters = list(dict_product(parameters))

    results_df_single = pd.DataFrame()

    with tqdm(parameters) as params:
        for iparams in params:
            # Update progress bar
            postfix = dict(
                Year=f"{iparams['cubes'][0]}",
                Temporal=f"{iparams['temporal']}",
                Spatial=f"{iparams['spatial']}",
            )
            params.set_postfix(postfix)

            # extract density cubes
            vod_df, lst_df, ndvi_df, sm_df = get_density_cubes(
                iparams["cubes"][1], iparams["spatial"], iparams["temporal"]
            )

            # get common elements
            dfs = get_common_elements_many([vod_df, lst_df, ndvi_df, sm_df])

            variables = {"VOD": dfs[0], "NDVI": dfs[1], "SM": dfs[2], "LST": dfs[3]}

            # get unique permutations
            res = set(
                tuple(
                    frozenset(sub)
                    for sub in set(
                        list(itertools.permutations(variables.keys(), args.compare))
                    )
                )
            )
            # do calculations for H, TC
            with tqdm(res) as iter_vars:
                for (ivar, jvar) in iter_vars:

                    prefix = dict(Variable1=f"{ivar}", Variable2=f"{jvar}",)
                    iter_vars.set_postfix(prefix)

                    # standardize data
                    X_norm = StandardScaler().fit_transform(variables[ivar])
                    Y_norm = StandardScaler().fit_transform(variables[jvar])

                    # Univariate statistics (pearson, spearman, kendall's tau)
                    uni_stats = univariate_stats(X_norm, Y_norm)

                    # entropy, total correlation
                    multivar_stats = get_similarity_scores(
                        X_norm, Y_norm, subsample=args.subsample,
                    )

                    # get H and TC
                    results_df_single = results_df_single.append(
                        {
                            "year": iparams["cubes"][0],
                            "drought": drought_years[str(iparams["cubes"][0])],
                            "samples": X_norm.shape[0],
                            "temporal": iparams["temporal"],
                            "variable1": ivar,
                            "variable2": jvar,
                            **multivar_stats,
                            **uni_stats,
                        },
                        ignore_index=True,
                    )

                    results_df_single.to_csv(SAVE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Drought Experiment.")

    # DataCube Arguments
    parser.add_argument(
        "--region", default="conus", type=str, help="The region for the drought events."
    )
    parser.add_argument(
        "--sampling",
        default="14D",
        type=str,
        help="The sampling scheme for drought events.",
    )

    # PreProcessing Arguments
    parser.add_argument(
        "--interp_method", default="linear", type=str, help="Interpolation method."
    )

    # Climatology Arguments
    parser.add_argument(
        "--climatology_window",
        default=2,
        type=int,
        help="Window length for climatology.",
    )
    parser.add_argument(
        "--subsample", type=int, default=10_000, help="subset points to take"
    )
    parser.add_argument(
        "-c", "--compare", type=int, default=2, help="variables to compare"
    )
    parser.add_argument(
        "-t",
        "--temporal",
        type=int,
        default=12,
        help="Max number of temporal dimensions",
    )
    parser.add_argument(
        "-s", "--spatial", type=int, default=1, help="Max number of spatial dimensions"
    )
    # logistics
    parser.add_argument(
        "--save", default="v0", type=str, help="Save Name for data results.",
    )
    main(parser.parse_args())
