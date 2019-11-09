import os, sys


import numpy as np

# Data Loaders
from src.data.climate.cmip5 import get_cmip5_model
from src.data.climate.amip import get_base_model
from src.data.climate.era5 import get_era5_data
from src.data.climate.ncep import get_ncep_data
from src.features.climate.build_features import (
    get_time_overlap,
    check_time_coords,
    regrid_2_lower_res,
    get_spatial_cubes,
    normalize_data,
)

from typing import Type, Union, Optional, Tuple, Dict

import xarray as xr


# Stat Tools
from src.models.train_models import run_rbig_models
from scipy import stats
import argparse

import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.utils import resample


xr_types = Union[xr.Dataset, xr.DataArray]


class DataArgs:
    # Path Arguments
    data_path = "/home/emmanuel/projects/2020_rbig_rs/data/climate/raw/amip/"
    results_path = "/home/emmanuel/projects/2020_rbig_rs/data/climate/results/amip/"


class CMIPArgs:
    # =============
    # Fixed Params
    # =============
    spatial_windows = list(range(1, 10 + 1))

    # ============
    # Free Params
    # ============
    variables = ["psl"]

    cmip_models = [
        "inmcm4",
        "access1_0",
        "bcc_csm1_1",
        "bcc_csm1_1_m",
        "bnu_esm",
        "giss_e2_r",
        "cnrm_cm5",
        "ipsl_cm5a_lr",
        "ipsl_cm5a_mr",
        "ipsl_cm5b_lr",
        "mpi_esm_lr",
        "mpi_esm_mr",
        "noresm1_m",
    ]

    base_models = ["ncep", "era5"]


def experiment_loop_comparative(
    base: str,
    cmip: str,
    variable: str,
    spatial_window: int,
    subsample: Optional[int] = None,
    trial: int = 1,
) -> Dict:
    """Performs one experimental loop for calculating the IT measures
    for the models"""
    # 1.1) get base model
    base_dat = get_base_model(base, variable)

    # 1.2) get cmip5 model
    cmip_dat = get_cmip5_model(cmip, variable)

    # 2) regrid data
    base_dat, cmip_dat = regrid_2_lower_res(base_dat, cmip_dat)

    # 3) find overlapping times
    base_dat, cmip_dat = get_time_overlap(base_dat, cmip_dat)

    # 4) get density cubes
    base_df = get_spatial_cubes(base_dat, spatial_window)
    cmip_df = get_spatial_cubes(cmip_dat, spatial_window)

    # 5) normalize data
    base_df = normalize_data(base_df)
    cmip_df = normalize_data(cmip_df)

    # 6) Resample/ Subsample data
    base_df = resample(base_df, n_samples=subsample, random_state=trial)
    cmip_df = resample(cmip_df, n_samples=subsample, random_state=trial)

    # 7.1) Mutual Information
    mutual_info, time_mi = run_rbig_models(base_df, cmip_df, measure="mi", verbose=None)

    # 7.2 - Pearson, Spearman, KendelTau
    pearson = stats.pearsonr(base_df.ravel(), cmip_df.ravel())[0]
    spearman = stats.spearmanr(base_df.ravel(), cmip_df.ravel())[0]
    kendelltau = stats.kendalltau(base_df.ravel(), cmip_df.ravel())[0]

    results = {
        "mi": mutual_info,
        "time_mi": time_mi,
        "pearson": pearson,
        "spearman": spearman,
        "kendelltau": kendelltau,
    }
    return results


def experiment_loop_individual(
    base: str,
    cmip: str,
    variable: str,
    spatial_window: int,
    subsample: Optional[int] = None,
    trial: int = 1,
    batch_size: Optional[int] = None,
) -> Dict:
    """Performs one experimental loop for calculating the IT measures
    for the models"""
    # 1.1) get base model
    base_dat = get_base_model(base, variable)

    # 1.2) get cmip5 model
    cmip_dat = get_cmip5_model(cmip, variable)

    # 2) regrid data
    base_dat, cmip_dat = regrid_2_lower_res(base_dat, cmip_dat)

    # 3) find overlapping times
    base_dat, cmip_dat = get_time_overlap(base_dat, cmip_dat)

    # 4) get density cubes
    base_df = get_spatial_cubes(base_dat, spatial_window)
    cmip_df = get_spatial_cubes(cmip_dat, spatial_window)

    # 5) normalize data
    base_df = normalize_data(base_df)
    cmip_df = normalize_data(cmip_df)

    # 6) Resample/ Subsample data
    base_df = resample(base_df, n_samples=subsample, random_state=trial)
    cmip_df = resample(cmip_df, n_samples=subsample, random_state=trial)

    # 7.1 - Entropy
    h_base, h_time_base = run_rbig_models(
        base_df, measure="h", verbose=None, batch_size=batch_size
    )
    h_cmip, h_time_cmip = run_rbig_models(
        cmip_df, measure="h", verbose=None, batch_size=batch_size
    )
    # 7.1 - Total Correlation
    tc_base, tc_time_base = run_rbig_models(
        base_df, measure="t", verbose=None, batch_size=batch_size
    )
    tc_cmip, tc_time_cmip = run_rbig_models(
        cmip_df, measure="t", verbose=None, batch_size=batch_size
    )

    results = {
        "h_base": h_base,
        "tc_base": tc_base,
        "h_cmip": h_cmip,
        "tc_cmip": tc_cmip,
        "t_base": h_time_base + tc_time_base,
        "t_cmip": h_time_cmip + tc_time_cmip,
    }
    return results


def experiment_individual(args):

    # initialize results
    results_df = pd.DataFrame()

    # set up progress bar
    n_iterations = len(CMIPArgs.cmip_models)
    with tqdm(list(range(n_iterations))) as pbar:

        # Loop through cmip models
        for i in pbar:
            icmip = CMIPArgs.cmip_models[i]

            # Loop through base models
            for ibase in CMIPArgs.base_models:

                # Loop through variables
                for ivariable in CMIPArgs.variables:

                    # Loop through spatial windows
                    for ispatial in CMIPArgs.spatial_windows:

                        # Get results
                        for itrial in range(args.trials):
                            ires = experiment_loop_individual(
                                ibase,
                                icmip,
                                ivariable,
                                ispatial,
                                args.subsample,
                                itrial,
                            )

                            # append results to running dataframe
                            results_df = results_df.append(
                                {
                                    "trial": itrial,
                                    "base": ibase,
                                    "cmip": icmip,
                                    "variable": ivariable,
                                    "spatial": ispatial,
                                    "h_base": ires["h_base"],
                                    "tc_base": ires["tc_base"],
                                    "h_cmip": ires["h_cmip"],
                                    "tc_cmip": ires["tc_cmip"],
                                    "t_base": ires["t_base"],
                                    "t_cmip": ires["t_cmip"],
                                    "subsample": args.subsample,
                                },
                                ignore_index=True,
                            )

                            # save results
                            results_df.to_csv(
                                DataArgs.results_path
                                + "global_individual_"
                                + args.save
                                + ".csv"
                            )
                            # Update Progress bar
                            postfix = dict(
                                Base=f"{ibase}",
                                CMIP=f"{icmip}",
                                Variable=f"{ivariable}",
                                Window=f"{ispatial}",
                            )
                            pbar.set_postfix(postfix)


def experiment_compare(args):
    # initialize results
    results_df = pd.DataFrame()

    # set up progress bar
    n_iterations = len(CMIPArgs.cmip_models)
    with tqdm(list(range(n_iterations))) as pbar:

        # Loop through cmip models
        for i in pbar:
            icmip = CMIPArgs.cmip_models[i]

            # Loop through base models
            for ibase in CMIPArgs.base_models:

                # Loop through variables
                for ivariable in CMIPArgs.variables:

                    # Loop through spatial windows
                    for ispatial in CMIPArgs.spatial_windows:

                        for itrial in range(args.trials):
                            # Get results
                            ires = experiment_loop_comparative(
                                ibase,
                                icmip,
                                ivariable,
                                ispatial,
                                args.subsample,
                                itrial,
                            )

                            # append results to running dataframe
                            results_df = results_df.append(
                                {
                                    "trial": itrial,
                                    "base": ibase,
                                    "cmip": icmip,
                                    "variable": ivariable,
                                    "spatial": ispatial,
                                    "mi": ires["mi"],
                                    "time_mi": ires["time_mi"],
                                    "pearson": ires["pearson"],
                                    "spearman": ires["spearman"],
                                    "kendelltau": ires["kendelltau"],
                                    "subsample": args.subsample,
                                },
                                ignore_index=True,
                            )

                            # save results
                            results_df.to_csv(
                                DataArgs.results_path
                                + "global_compare_"
                                + args.save
                                + ".csv"
                            )
                            # Update Progress bar
                            postfix = dict(
                                Trial=f"{itrial}",
                                Base=f"{ibase}",
                                CMIP=f"{icmip}",
                                Variable=f"{ivariable}",
                                Window=f"{ispatial}",
                            )
                            pbar.set_postfix(postfix)


def main(args):

    if args.exp == "individual":
        experiment_individual(args)

    elif args.exp == "compare":
        experiment_compare(args)

    else:
        raise ValueError("Unrecognized experiment:", args.exp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Arguments for climate model global experiment"
    )

    parser.add_argument(
        "--exp",
        default="individual",
        type=str,
        help="Individual IT measures or comparative IT measures.",
    )

    # ===================
    # IT Measures Params
    # ===================
    parser.add_argument(
        "--subsample", default=None, type=int, help="Whether to do a subsample"
    )

    parser.add_argument(
        "--ensemble",
        default=False,
        type=bool,
        help="Whether to do an ensemble with some subsample.",
    )

    parser.add_argument(
        "--save", default="v0", type=str, help="Save name for experiment."
    )
    parser.add_argument("--trials", default=1, type=int, help="Number of trials")
    # Parse Arguments and run experiment
    main(parser.parse_args())
