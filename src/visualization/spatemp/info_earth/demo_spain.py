import sys, os
from pyprojroot import here
from typing import Optional

root = here(project_files=[".here"])
sys.path.append(str(here()))

import pathlib

# standard python packages
import xarray as xr

# NUMPY SETTINGS
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# MATPLOTLIB Settings
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs


# SEABORN SETTINGS
import seaborn as sns

sns.set_context(context="talk", font_scale=0.7)
# sns.set(rc={'figure.figsize': (12, 9.)})
# sns.set_style("whitegrid")

# PANDAS SETTINGS
import pandas as pd

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)


RES_PATH = pathlib.Path(str(root)).joinpath("data/spa_temp/info_earth")
FIG_PATH = pathlib.Path(str(root)).joinpath(
    "reports/figures/spa_temp/demos/infoearth/spain"
)


def plot_map(xr_data, measure: str, save_name: Optional[str] = None):
    fig, ax = plt.subplots()

    if measure == "probs":
        xr_data.probs.mean(dim="time").plot(
            ax=ax,
            vmin=0,
            robust=True,
            cmap="Reds",
            cbar_kwargs={"label": "Probability"},
        )
    elif measure == "info":
        xr_data.shannon_info.mean(dim="time").plot(
            ax=ax,
            vmin=0,
            robust=True,
            cmap="Reds",
            cbar_kwargs={"label": "Shannon Information"},
        )
    else:
        raise ValueError(f"Unrecognized measure: {measure}")

    ax.set(xlabel="Longitude", ylabel="Latitude")
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_PATH.joinpath(f"{measure}_maps_{save_name}.png"))


def plot_ts(xr_data, measure: str, save_name: Optional[str] = None):
    fig, ax = plt.subplots()

    if measure == "probs":
        xr_data.probs.mean(dim=["lon", "lat"]).plot.line(
            ax=ax, color="black", linewidth=3
        )
        ylabel = "Probability"
    elif measure == "info":
        xr_data.shannon_info.mean(dim=["lon", "lat"]).plot.line(
            ax=ax, color="black", linewidth=3
        )
        ylabel = "Shannon Information"
    else:
        raise ValueError(f"Unrecognized measure: {measure}")

    ax.set(xlabel="Time", ylabel=ylabel)

    ax.legend(["Mean Predictions"])
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_PATH.joinpath(f"{measure}_ts_{save_name}.png"))


def plot_ts_error(xr_data, measure: str, save_name: Optional[str] = None):

    if measure == "probs":
        predictions = xr_data.probs.mean(dim=["lat", "lon"])
        std = xr_data.probs.std(dim=["lat", "lon"])
        ylabel = "Probabilities"
    elif measure == "info":
        predictions = xr_data.shannon_info.mean(dim=["lat", "lon"])
        std = xr_data.shannon_info.std(dim=["lat", "lon"])
        ylabel = "Shannon Information"
    else:
        raise ValueError(f"Unrecognized measure: {measure}")

    fig, ax = plt.subplots()
    ax.plot(xr_data.coords["time"].values, predictions)
    ax.fill_between(
        predictions.coords["time"].values,
        predictions - std,
        predictions + std,
        alpha=0.7,
        color="orange",
    )
    ax.set(
        xlabel="Time", ylabel=ylabel,
    )
    ax.legend(["Mean_predictions"])
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_PATH.joinpath(f"{measure}_ts_err_{save_name}.png"))


def plot_monthly_map(xr_data, measure: str, save_name: Optional[str] = None):
    plt.figure()
    xr_data.probs.groupby("time.month").mean().plot.pcolormesh(
        x="lon", y="lat", col="month", col_wrap=3, vmin=0, robust=True, cmap="Reds"
    )
    plt.savefig(FIG_PATH.joinpath(f"{measure}_monthly_{save_name}.png"))


if __name__ == "__main__":

    region = "spain"
    dimensions = ["111", "116", "331", "333"]
    for idimension in dimensions:
        period = "2002_2010"
        samples = "200000"

        variable = "gpp"
        filename = f"{region}_{variable}_{period}_v0_s{samples}_d{idimension}"

        # read csv file
        probs_df = pd.read_csv(str(RES_PATH.joinpath(f"probs/{filename}" + ".csv")))

        # convert to datetime
        probs_df["time"] = pd.to_datetime(probs_df["time"])

        # create dataframe in the format for xarray
        probs_df = probs_df.set_index(["time", "lat", "lon"]).rename(
            columns={"0": "probs"}
        )

        # remove probabilities greater than 1
        probs_df["probs"][probs_df["probs"] >= 1.0] = np.nan

        # shannon info
        probs_df["shannon_info"] = -np.log(probs_df["probs"])

        # create xarray cubes
        probs_cubes = xr.Dataset.from_dataframe(probs_df)

        # Probability / Shannon Information Maps
        plot_map(probs_cubes, "probs", f"{filename}")
        plot_map(probs_cubes, "info", f"{filename}")

        # Probability Maps (per month)
        plot_monthly_map(probs_cubes, "probs", f"{filename}")
        plot_monthly_map(probs_cubes, "info", f"{filename}")

        plot_ts(probs_cubes, "probs", f"{filename}")
        plot_ts(probs_cubes, "info", f"{filename}")

        plot_ts_error(probs_cubes, "probs", f"{filename}")
        plot_ts_error(probs_cubes, "info", f"{filename}")
