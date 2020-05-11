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


RES_PATH = pathlib.Path(str(root)).joinpath("data/spa_temp/info_earth/entropy")
FIG_PATH = pathlib.Path(str(root)).joinpath(
    "reports/figures/spa_temp/demos/infoearth/spain"
)

from typing import Optional


def plot_entropies(df, save_name: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.lineplot(
        x="ratio",
        y="entropy_norm",
        hue="n_dimensions",
        data=df,
        ax=ax,
        marker=".",
        markersize=20,
        palette="cubehelix_r",
    )
    ax.set(
        xlabel="Spatial / Temporal Ratio", ylabel="Entropy", xscale="log",
    )
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_PATH.joinpath(f"H_{save_name}.png"))


def main():

    region = "spain"
    period = "2010"
    variables = ["gpp", "rm", "precip", "lst", "sm"]
    for ivariable in variables:
        save_name = f"{region}_{ivariable}_{period}"

        # read csv file
        results_df = pd.read_csv(str(RES_PATH.joinpath(f"v0_{save_name}.csv")))
        results_df["n_dimensions"][results_df["n_dimensions"] == 46] = 49

        sub_df = results_df.copy()

        sub_df["ratio"] = sub_df["spatial"] ** 2 / sub_df["temporal"]
        sub_df["entropy_norm"] = sub_df["entropy"] / (
            sub_df["spatial"] ** 2 * sub_df["temporal"]
        )
        # sub_df['dimensions'] = sub_df['spatial'] ** 2 + sub_df['temporal']

        # sub_df['nH'] = (sub_df['rbig_H_y'] /  sub_df['dimensions']) #* np.log(2)
        sub_df["spatial"] = sub_df["spatial"].astype("category")
        sub_df["temporal"] = sub_df["temporal"].astype("category")
        sub_df["n_dimensions"] = sub_df["n_dimensions"].astype("category")

        plot_entropies(sub_df, save_name)


if __name__ == "__main__":
    main()
