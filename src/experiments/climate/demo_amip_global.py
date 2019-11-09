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
from scipy import stats


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

    # 7.1) Mutual Information
    print("Mutual Information")
    mutual_info, time_mi = run_rbig_models(
        base_df[:subsample],
        cmip_df[:subsample],
        measure="mi",
        verbose=1,
        batch_size=batch_size,
    )

    # 7.2 - Pearson, Spearman, KendelTau
    print("Pearson")
    pearson = stats.pearsonr(base_df[:subsample].ravel(), cmip_df[:subsample].ravel())[
        0
    ]
    print("Spearman")
    spearman = stats.spearmanr(
        base_df[:subsample].ravel(), cmip_df[:subsample].ravel()
    )[0]
    print("KendelTau")
    kendelltau = stats.kendalltau(
        base_df[:subsample].ravel(), cmip_df[:subsample].ravel()
    )[0]

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

    # 7.1) Total Correlation
    print("TOTAL CORRELATION")
    print("Base Model")
    tc_base, time_base = run_rbig_models(
        base_df[:subsample], measure="t", verbose=1, batch_size=batch_size
    )
    print("CMIP Model")
    tc_cmip, time_cmip = run_rbig_models(
        cmip_df[:subsample], measure="t", verbose=1, batch_size=batch_size
    )

    # 7.2 - ENTROPY
    print("ENTROPY")
    print("Base Model")
    h_base, time_base = run_rbig_models(
        base_df[:subsample], measure="h", verbose=1, batch_size=batch_size
    )
    print("CMIP Model")
    h_cmip, time_cmip = run_rbig_models(
        cmip_df[:subsample], measure="h", verbose=1, batch_size=batch_size
    )

    results = {
        "h_base": h_base,
        "tc_base": tc_base,
        "h_cmip": h_cmip,
        "tc_cmip": tc_cmip,
        "t_base": time_base,
        "t_cmip": time_cmip,
    }
    return results


def experiment_individual(args):

    i = 0

    results_df = pd.DataFrame()

    icmip = CMIPArgs.cmip_models[i]
    print("CMIP Model:", icmip)

    # get base model
    ibase = CMIPArgs.base_models[i]
    print("Base Model:", ibase)

    # loop through variables
    ivariable = CMIPArgs.variables[i]
    print("Variable:", ivariable)

    # get spatial window
    ispatial = CMIPArgs.spatial_windows[i]
    ispatial = 3
    print("Spatial Window:", ispatial)

    # get results
    iresult = experiment_loop_individual(
        ibase, icmip, ivariable, ispatial, args.subsample, args.batchsize
    )
    print(iresult)

    # append to running dataframe
    results_df = results_df.append(
        {
            "base": ibase,
            "cmip": icmip,
            "variable": ivariable,
            "spatial": ispatial,
            "h_base": iresult["h_base"],
            "tc_base": iresult["tc_base"],
            "h_cmip": iresult["h_cmip"],
            "tc_cmip": iresult["tc_cmip"],
            "t_base": iresult["t_base"],
            "t_cmip": iresult["t_cmip"],
            "subsample": args.subsample,
        },
        ignore_index=True,
    )

    results_df.to_csv(DataArgs.results_path + "demo_individual_" + args.save + ".csv")


def experiment_compare(args):

    i = 0

    results_df = pd.DataFrame()

    icmip = CMIPArgs.cmip_models[i]
    print("CMIP Model:", icmip)

    # get base model
    ibase = CMIPArgs.base_models[i]
    print("Base Model:", ibase)

    # loop through variables
    ivariable = CMIPArgs.variables[i]
    print("Variable:", ivariable)

    # get spatial window
    ispatial = CMIPArgs.spatial_windows[i]
    ispatial = 3
    print("Spatial Window:", ispatial)

    # get results
    iresult = experiment_loop_comparative(
        ibase, icmip, ivariable, ispatial, args.subsample, args.batchsize
    )
    print(iresult)

    # append results to running dataframe
    results_df = results_df.append(
        {
            "base": ibase,
            "cmip": icmip,
            "variable": ivariable,
            "spatial": ispatial,
            "mi": iresult["mi"],
            "time_mi": iresult["time_mi"],
            "pearson": iresult["pearson"],
            "spearman": iresult["spearman"],
            "kendelltau": iresult["kendelltau"],
            "subsample": args.subsample,
        },
        ignore_index=True,
    )
    # save results
    results_df.to_csv(DataArgs.results_path + "demo_compare_" + args.save + ".csv")


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
        "--subsample", default=10_000, type=int, help="Whether to do a subsample"
    )
    parser.add_argument(
        "--batchsize", default=None, type=int, help="How many batches to generate"
    )

    parser.add_argument(
        "--ensemble",
        default=False,
        type=bool,
        help="Whether to do an ensemble with some subsample.",
    )

    parser.add_argument(
        "--save", default="demo_v1", type=str, help="Save name for experiment."
    )
    parser.add_argument("--trials", default=10, type=int, help="Number of trials")

    # Parse Arguments and run experiment
    main(parser.parse_args())
