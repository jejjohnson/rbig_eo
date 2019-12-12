import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

plt.style.use(["seaborn-talk", "ggplot"])

SAVEPATH = (
    "/home/emmanuel/projects/2020_rbig_rs/reports/figures/climate/amip/global/entropy/"
)


def plot_global_entropy(
    results_df: pd.DataFrame,
    base: str,
    cmip: str,
    normalized=True,
    log_mi=True,
    save=True,
) -> None:

    if normalized == True:
        results_df["h_cmip"] = results_df["h_cmip"] / results_df["spatial"] ** 2
        results_df["h_base"] = results_df["h_base"] / results_df["spatial"] ** 2

    # subset
    results_df = results_df[results_df["base"] == base]
    results_df = results_df[results_df["cmip"] == cmip]
    fig, ax = plt.subplots()
    sns.scatterplot(data=results_df, x="spatial", y="h_base", label=f"{base}", ax=ax)
    sns.scatterplot(data=results_df, x="spatial", y="h_cmip", label=f"{cmip}", ax=ax)
    ax.set_title(f"{base.upper()} vs CMIP: {cmip.upper()}")
    ax.set_xlabel("Spatial Features")
    #     ax.set_xlim([2, 6])
    ax.set_ylabel("Entropy, H")
    ax.legend()

    if save:
        savename = f"global_h_{base}_{cmip}.png"
        fig.savefig(SAVEPATH + savename)
    else:
        plt.show()
    return None


def plot_global_diff_entropy(
    results_df: pd.DataFrame,
    base: str,
    normalized: bool = True,
    log_mi: bool = False,
    save: bool = True,
) -> None:

    # print(results_df.head())
    results_copy = results_df[results_df["base"] == base]
    # print(results_copy.head())
    if normalized == True:
        results_copy["h_base"] = results_copy["h_base"] / results_copy["spatial"] ** 2
        results_copy["h_cmip"] = results_copy["h_cmip"] / results_copy["spatial"] ** 2

    # print(results_copy.head())
    # calculate difference
    results_copy["h_diff"] = np.abs(results_copy["h_cmip"] - results_copy["h_base"])
    # print(results_copy.head())
    if log_mi == True:
        results_copy["h_diff"] = np.log(results_copy["h_diff"])

    fig, ax = plt.subplots()
    #     sns.scatterplot(ax=ax, data=results_copy, x='spatial', y='h_diff', hue='base', color='black')
    # print(results_copy.head())
    sns.lineplot(
        ax=ax,
        data=results_copy,
        x="spatial",
        y="h_diff",
        hue="cmip",
        linewidth=6,
        marker="o",
    )

    plt.title(f"")
    plt.xlabel("Spatial Features", fontsize=20)
    plt.ylabel("Difference in Entropy", fontsize=20)
    plt.legend(ncol=2, bbox_to_anchor=(2.05, 1), fontsize=16)
    if save:
        savename = f"global_dh_{base}.png"
        fig.savefig(SAVEPATH + savename)
    else:
        plt.show()
    return None


def plot_global_mutual_info(
    results_df: pd.DataFrame,
    base: str,
    measure: str,
    cmip: Optional[str] = None,
    normalized=True,
    log_mi=True,
    save=True,
) -> None:

    # subset
    results_df = results_df[results_df["base"] == base]
    if cmip is not None:
        results_df = results_df[results_df["cmip"] == cmip]

    if normalized and measure == "mi":
        # print("norm")
        results_df[measure] = results_df[measure] / results_df["spatial"] ** 2
    # print(results_df.head())
    if log_mi == True:
        results_df[measure] = np.log10(1 + results_df[measure])
    fig, ax = plt.subplots()
    sns.lineplot(data=results_df, x="spatial", y=measure, hue="cmip", ax=ax)

    # ax.set_title(f"{base.upper()} vs CMIP: {cmip.upper()}")
    ax.set_xlabel("Spatial Features")
    #     ax.set_xlim([2, 6])
    ax.set_ylabel("Mutual Information")
    ax.legend()

    if save:
        savename = f"global_mi_{base}_{cmip}.png"
        fig.savefig(SAVEPATH + savename)
    else:
        plt.show()
    return None


def plot_global_diff_mutual_info(
    results_df: pd.DataFrame,
    base: str,
    normalized: bool = True,
    log_mi: bool = False,
    save: bool = True,
) -> None:

    # print(results_df.head())
    results_copy = results_df[results_df["base"] == base]
    # print(results_copy.head())
    if normalized == True:
        results_copy["h_base"] = results_copy["h_base"] / results_copy["spatial"] ** 2
        results_copy["h_cmip"] = results_copy["h_cmip"] / results_copy["spatial"] ** 2

    # print(results_copy.head())
    # calculate difference
    results_copy["h_diff"] = np.abs(results_copy["h_cmip"] - results_copy["h_base"])
    # print(results_copy.head())
    if log_mi == True:
        results_copy["h_diff"] = np.log(results_copy["h_diff"])

    fig, ax = plt.subplots()
    #     sns.scatterplot(ax=ax, data=results_copy, x='spatial', y='h_diff', hue='base', color='black')
    # print(results_copy.head())
    sns.lineplot(
        ax=ax,
        data=results_copy,
        x="spatial",
        y="h_diff",
        hue="cmip",
        linewidth=6,
        marker="o",
    )

    plt.title(f"")
    plt.xlabel("Spatial Features", fontsize=20)
    plt.ylabel("Difference in Entropy", fontsize=20)
    plt.legend(ncol=2, bbox_to_anchor=(2.05, 1), fontsize=16)
    if save:
        savename = f"global_dh_{base}.png"
        fig.savefig(SAVEPATH + savename)
    else:
        plt.show()
    return None
