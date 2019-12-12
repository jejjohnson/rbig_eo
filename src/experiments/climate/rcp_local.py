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
    get_reference_dataset,
    regrid_data,
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
from sklearn.utils import resample
from scipy import stats

import warnings

warnings.simplefilter("ignore")

xr_types = Union[xr.Dataset, xr.DataArray]


class DataArgs:
    # Path Arguments
    data_path = "/home/emmanuel/projects/2020_rbig_rs/data/climate/raw/rcp/"
    interim_path = "/home/emmanuel/projects/2020_rbig_rs/data/climate/interim/rcp/"
    results_path = "/home/emmanuel/projects/2020_rbig_rs/data/climate/results/rcp/"


class CMIPArgs:
    # =============
    # Fixed Params
    # =============
    spatial_windows = [1, 2, 3, 4, 5]

    # ============
    # Free Params
    # ============
    variables = ["psl"]

    cmip_models = [
        "inmcm4",
        "access1_0",
        "access1_3",
        "ipsl_cm5a_mr",
        "mpi_esm_lr",
        "mpi_esm_mr",
        "noresm1_m",
    ]

    base_models = ["ncep", "era5"]


def get_features_loop(base: str, cmip: str, variable: str, regrid_name: str) -> Dict:
    """Performs one experimental loop for calculating the IT measures
    for the models"""
    # 1.1) get base model
    # print(base)
    base_dat = get_base_model(base, variable)

    # 1.2) get cmip5 model
    cmip_dat = get_cmip5_model(cmip, variable, model="rcp")

    # 2) regrid data

    # 2) regrid data w. reference grid and
    reference_ds = get_reference_dataset("noresm1_m")
    base_dat = regrid_data(reference_ds, base_dat, filename=regrid_name + ".nc")
    cmip_dat = regrid_data(reference_ds, cmip_dat, filename=regrid_name + ".nc")

    # 3) find overlapping times
    base_dat, cmip_dat = get_time_overlap(base_dat, cmip_dat)

    return base_dat, cmip_dat


def generate_temporal_data(base_dat, cmip_dat, time: Optional[str] = "month"):

    if time == "month":
        time_stamps = min(len(base_dat.time), len(cmip_dat.time))
        for itime in range(time_stamps):
            itime_stamp = base_dat.time.values
            ibase_dat = base_dat.isel(time=itime)
            icmip_dat = cmip_dat.isel(time=itime)
            ibase_dat = ibase_dat.expand_dims({"time": 1})
            icmip_dat = icmip_dat.expand_dims({"time": 1})
            yield ibase_dat, icmip_dat
    elif time == "year":
        base_dat = base_dat.groupby("time.year")
        cmip_dat = cmip_dat.groupby("time.year")
        for ibase_dat, icmip_dat in zip(base_dat, cmip_dat):
            yield ibase_dat[1], icmip_dat[1]


def process_loop_individual(
    base_dat: str,
    cmip_dat: str,
    spatial_window: int,
    subsample: Optional[int] = None,
    trial: int = 1,
    batch_size: Optional[int] = None,
) -> Dict:

    # get time stamp
    base_time_stamp = base_dat.time.values[0]
    cmip_time_stamp = cmip_dat.time.values[0]

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
    tc_base, h_base, h_time_base = run_rbig_models(base_df, measure="h", verbose=None)
    tc_cmip, h_cmip, h_time_cmip = run_rbig_models(cmip_df, measure="h", verbose=None)

    results = {
        "base_time_stamp": base_time_stamp,
        "cmip_time_stamp": cmip_time_stamp,
        "h_base": h_base,
        "h_cmip": h_cmip,
        "tc_base": tc_base,
        "tc_cmip": tc_cmip,
        "t_base": h_time_base,
        "t_cmip": h_time_cmip,
    }
    return results


def process_loop_compare(
    base_dat: str,
    cmip_dat: str,
    spatial_window: int,
    subsample: Optional[int] = None,
    trial: int = 1,
) -> Dict:
    """Performs one experimental loop for calculating the IT measures
    for the models"""
    # 1.1) get base model
    # get time stamp
    base_time_stamp = base_dat.time.values[0]
    cmip_time_stamp = cmip_dat.time.values[0]

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
        "base_time_stamp": base_time_stamp,
        "cmip_time_stamp": cmip_time_stamp,
        "mi": mutual_info,
        "time_mi": time_mi,
        "pearson": pearson,
        "spearman": spearman,
        "kendelltau": kendelltau,
    }
    return results


def experiment_individual(args):

    # initialize results
    results_df = pd.DataFrame()

    # cmip_model
    base_model = CMIPArgs.base_models[args.base]
    cmip_model = CMIPArgs.cmip_models[args.cmip]
    interim_name = (
        DataArgs.interim_path
        + "local/individual/"
        + f"{base_model}_{cmip_model}_"
        + args.save
    )
    results_name = (
        DataArgs.results_path
        + "local/individual/"
        + f"{base_model}_{cmip_model}_"
        + args.save
    )

    # set up progress bar
    n_iterations = len(CMIPArgs.cmip_models)
    with tqdm(CMIPArgs.spatial_windows) as pbar:

        # Loop through cmip models
        for ispatial in pbar:

            # Loop through variables
            for ivariable in CMIPArgs.variables:

                # Get results
                base_dat, cmip_dat = get_features_loop(
                    base_model, cmip_model, ivariable, regrid_name=interim_name
                )

                # generate dataset per year
                for ibase_dat, icmip_dat in generate_temporal_data(
                    base_dat, cmip_dat, time=args.time
                ):
                    for itrial in range(args.trials):
                        # generate temporal data
                        ires = process_loop_individual(
                            ibase_dat, icmip_dat, ispatial, args.subsample, itrial
                        )

                        # append results to running dataframe
                        results_df = results_df.append(
                            {
                                "trial": itrial,
                                "base": base_model,
                                "cmip": cmip_model,
                                "variable": ivariable,
                                "spatial": ispatial,
                                "base_time": ires["base_time_stamp"],
                                "cmip_time": ires["cmip_time_stamp"],
                                "h_base": ires["h_base"],
                                "h_cmip": ires["h_cmip"],
                                "tc_cmip": ires["tc_cmip"],
                                "tc_base": ires["tc_base"],
                                "t_base": ires["t_base"],
                                "t_cmip": ires["t_cmip"],
                                "subsample": args.subsample,
                            },
                            ignore_index=True,
                        )

                        # save results
                        results_df.to_csv(interim_name + ".csv")
                        # Update Progress bar
                        postfix = dict(
                            Base=f"{base_model}",
                            CMIP=f"{cmip_model}",
                            Variable=f"{ivariable}",
                            Window=f"{ispatial}",
                            Year=ires["base_time_stamp"],
                        )
                        pbar.set_postfix(postfix)
    results_df.to_csv(results_name + ".csv")


def experiment_compare(args):
    # initialize results
    results_df = pd.DataFrame()

    # cmip_model
    base_model = CMIPArgs.base_models[args.base]
    cmip_model = CMIPArgs.cmip_models[args.cmip]

    interim_name = (
        DataArgs.interim_path
        + "local/compare/"
        + f"{base_model}_{cmip_model}_"
        + args.save
    )
    results_name = (
        DataArgs.results_path
        + "local/compare/"
        + f"{base_model}_{cmip_model}_"
        + args.save
    )
    # set up progress bar
    n_iterations = len(CMIPArgs.cmip_models)
    with tqdm(CMIPArgs.spatial_windows) as pbar:

        # Loop through cmip models
        for ispatial in pbar:

            # Loop through variables
            for ivariable in CMIPArgs.variables:

                # Get results
                base_dat, cmip_dat = get_features_loop(
                    base_model, cmip_model, ivariable, interim_name
                )

                # generate dataset per year
                for ibase_dat, icmip_dat in generate_temporal_data(
                    base_dat, cmip_dat, time=args.time
                ):
                    for itrial in range(args.trials):

                        # generate temporal data
                        ires = process_loop_compare(
                            ibase_dat, icmip_dat, ispatial, args.subsample, itrial
                        )

                        # append results to running dataframe
                        results_df = results_df.append(
                            {
                                "trial": itrial,
                                "base": base_model,
                                "cmip": cmip_model,
                                "variable": ivariable,
                                "spatial": ispatial,
                                "base_time": ires["base_time_stamp"],
                                "cmip_time": ires["cmip_time_stamp"],
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

                        results_df.to_csv(interim_name + ".csv")

                        # Update Progress bar
                        postfix = dict(
                            Trial=f"{itrial}",
                            Base=f"{base_model}",
                            CMIP=f"{cmip_model}",
                            Variable=f"{ivariable}",
                            Window=f"{ispatial}",
                            Year=ires["base_time_stamp"],
                        )
                        pbar.set_postfix(postfix)

    results_df.to_csv(results_name + ".csv")


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

    # ======================
    # Experiment Parameters
    # ======================
    parser.add_argument(
        "--exp",
        default="individual",
        type=str,
        help="Individual IT measures or comparative IT measures.",
    )
    parser.add_argument(
        "--base", default=0, type=int, help="Base model to be compared (ncep, era5)"
    )
    parser.add_argument(
        "--cmip", default=0, type=int, help="CMIP model to be compared (inmcm4)"
    )
    parser.add_argument("--time", default="year", type=str, help="The time to group.")
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
        "--save", default="v2", type=str, help="Save name for experiment."
    )
    parser.add_argument("--trials", default=10, type=int, help="Number of trials")

    # Parse Arguments and run experiment
    main(parser.parse_args())
