import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from pyprojroot import here
import pathlib

PATH = pathlib.PATH(str(here()))
# root = here(project_files=[".here"])
sys.path.append(str(here()))
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# drought tools
from src.data.drought.loader import DataLoader
from src.experiments.utils import dict_product
from src.features.drought.build_features import (
    get_cali_geometry,
    get_common_elements_many,
    get_density_cubes,
    mask_datacube,
    remove_climatology,
)
from src.models.similarity import rbig_h_measures

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


DATA_PATH = "/home/emmanuel/projects/2020_rbig_rs/data/drought/results/"

DROUGHT_YEARS = {
    "2010": False,
    "2011": False,
    "2012": True,
    "2013": False,
    "2014": True,
    "2015": True,
}


def main(args):

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

    results_df_single = pd.DataFrame()

    # group datacube by years
    # logger.info(
    #     f"Starting loop through {len(time_steps)} months and {len(drought_years)} years."
    # )
    parameters = {}
    parameters["cubes"] = list(drought_cube.groupby("time.year"))
    parameters["temporal"] = np.arange(1, 12)
    parameters["spatial"] = [1]

    parameters = list(dict_product(parameters))

    with tqdm(parameters) as params:
        for iparams in params:

            # extract density cubes
            vod_df, lst_df, ndvi_df, sm_df = get_density_cubes(
                iparams["cubes"][1], iparams["spatial"], iparams["temporal"]
            )

            # get common elements
            dfs = get_common_elements_many([vod_df, lst_df, ndvi_df, sm_df])
            vod_df, lst_df, ndvi_df, sm_df = dfs[0], dfs[1], dfs[2], dfs[3]

            variables = {"VOD": vod_df, "NDVI": ndvi_df, "SM": sm_df, "LST": lst_df}

            # do calculations for H, TC
            for iname, idata in variables.items():

                # normalize data
                X_norm = StandardScaler().fit_transform(idata)

                # ==================
                # Algorithms
                # =================

                # RV Coefficient

                # CKA Coefficient

                # IT Measures (RBIG)
                t0 = time.time()
                rbig_h = rbig_h_measures(X_norm, subsample=None, random_state=123)
                t1 = time.time() - t0
                # get H and TC
                results_df_single = results_df_single.append(
                    {
                        "year": iparams["cubes"][0],
                        "drought": drought_years[str(iparams["cubes"][0])],
                        "samples": X_norm.shape[0],
                        "temporal": iparams["temporal"],
                        "variable": iname,
                        "h": rbig_h,
                        "time": t1,
                    },
                    ignore_index=True,
                )

                results_df_single.to_csv(DATA_PATH + args.save)
                postfix = dict(
                    Year=f"{iparams['cubes'][0]}",
                    Temporal=f"{iparams['temporal']}",
                    Spatial=f"{iparams['spatial']}",
                    Variable=f"{iname}",
                )
                params.set_postfix(postfix)


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

    # logistics
    parser.add_argument(
        "--save", default="exp_ind_v0.csv", type=str, help="Save Name for data results."
    )
    main(parser.parse_args())
