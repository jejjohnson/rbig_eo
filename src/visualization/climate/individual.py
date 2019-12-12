import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use(["seaborn-talk", "ggplot"])

FIG_PATH = "/home/emmanuel/projects/2020_rbig_rs/reports/figures/"


def plot_individual(
    df: pd.DataFrame,
    cmip_model: str,
    spatial_res: int,
    info: str = "h",
    model: str = "amip",
    save: bool = False,
) -> None:

    # subset data
    df = df[df["cmip"] == cmip_model]
    df = df[df["spatial"] == spatial_res]

    if model == "amip":
        xticks = np.arange(1980, 2009, 1)
    elif model == "rcp":
        xticks = np.arange(1979, 2019, 1)
    else:
        raise ValueError("Unrecognized model:", model)

    fig, ax = plt.subplots(figsize=(10, 7))

    if info == "h":
        pts1 = sns.lineplot(x="base_time", y="h_base", data=df, linewidth=5, marker="o")
        pts2 = sns.lineplot(x="base_time", y="h_cmip", data=df, linewidth=5, marker="o")
        ax.set_xticklabels(xticks, fontsize=20)
        ax.set_ylabel("Entropy, H")
    elif info == "mi":
        pts1 = sns.lineplot(x="cmip_time", y="mi", data=df, linewidth=5)
        ax.set_xticklabels(xticks, fontsize=20)
        ax.set_ylabel("Mutual Information, MI")
    else:
        raise ValueError("Unrecognized info measure:", info)
    plt.xticks(rotation="vertical")
    ax.set_xlabel("")
    if save:
        fig.savefig()
    else:
        plt.show()


def plot_individual_all(
    df: pd.DataFrame, spatial_res: int, info: str = "h", model: str = "amip"
) -> None:

    # subset data
    df = df[df["spatial"] == spatial_res]

    if model == "amip":
        xticks = np.arange(1980, 2009, 1)
    elif model == "rcp":
        xticks = np.arange(1979, 2019, 1)
    else:
        raise ValueError("Unrecognized model:", model)

    fig, ax = plt.subplots(figsize=(10, 7))
    if info == "h":
        pts1 = sns.lineplot(
            x="base_time",
            y="h_base",
            data=df,
            linestyle="--",
            color="black",
            linewidth=6,
        )
        pts2 = sns.lineplot(
            x="base_time", y="h_cmip", data=df, hue="cmip", linewidth=5, marker="o"
        )
        ax.set_ylabel("Entropy, H")
        ax.set_xticklabels(xticks, fontsize=20)
        ax.set_xlims([xticks[0], xticks[-1]])
    elif info == "mi":
        pts2 = sns.lineplot(
            x="cmip_time", y="mi", data=df, hue="cmip", linewidth=5, marker="o"
        )
        ax.set_xticklabels(xticks, fontsize=20)
        ax.set_xlims([xticks[0], xticks[-1]])
        ax.set_ylabel("Mutual Information, MI")
    else:
        raise ValueError("Unrecognized info measure:", info)
    plt.xticks(rotation="vertical")
    ax.set_xlabel("")
    plt.show()


def plot_diff(df: pd.DataFrame, spatial_res: int) -> None:

    # subset data
    df = df[df["spatial"] == spatial_res]

    fig, ax = plt.subplots(figsize=(10, 7))

    df["h_abs_diff"] = abs(df["h_base"] - df["h_cmip"])

    pts2 = sns.lineplot(x="base_time", y="h_abs_diff", hue="cmip", data=df, linewidth=5)
    ax.set_xlabel("Time", fontsize=20)
    ax.set_xticklabels(np.arange(1980, 2009, 1), fontsize=20)
    ax.set_ylabel("Absolute Difference, H", fontsize=20)
    plt.xticks(rotation="vertical")
    plt.show()


def plot_individual_diff(df: pd.DataFrame, cmip_model: str, spatial_res: int) -> None:

    # subset data
    df = df[df["cmip"] == cmip_model]
    df = df[df["spatial"] == spatial_res]

    fig, ax = plt.subplots(figsize=(10, 7))

    df["h_abs_diff"] = abs(df["h_base"] - df["h_cmip"])

    pts2 = sns.lineplot(x="base_time", y="h_abs_diff", data=df, linewidth=5)
    plt.xticks(rotation="vertical")
    ax.set_xticklabels(np.arange(1980, 2009, 1), fontsize=20)
    plt.show()
