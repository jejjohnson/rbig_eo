import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use(["seaborn-talk", "ggplot"])


FIG_PATH = "/home/emmanuel/projects/2020_rbig_rs/reports/figures/climate/"


def plot_individual(
    df: pd.DataFrame,
    base_model: str,
    cmip_model: str,
    spatial_res: int,
    info: str = "h",
    model: str = "amip",
    save: bool = False,
) -> None:

    # subset data
    df = df[df["base"] == base_model]
    df = df[df["cmip"] == cmip_model]
    df = df[df["spatial"] == spatial_res]

    if model == "amip":
        xticks = np.arange(1980, 2009, 1)
    elif model == "rcp":
        xticks = np.arange(1980, 2019, 1)
    else:
        raise ValueError("Unrecognized model:", model)

    fig, ax = plt.subplots(figsize=(10, 7))

    if info == "h":
        pts1 = sns.lineplot(x="base_time", y="h_base", data=df, linewidth=5)
        pts2 = sns.lineplot(x="base_time", y="h_cmip", data=df, linewidth=5)
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
        fig.savefig(f"{FIG_PATH}/{model}/local/{info}_{cmip_model}.png")
    else:
        plt.show()


def plot_individual_all(
    df: pd.DataFrame,
    base_model: str,
    spatial_res: int,
    info: str = "h",
    model: str = "amip",
    save: bool = False,
) -> None:

    palette = {
        "ipsl_cm5a_mr": "blue",
        "ipsl_cm5a_lr": "blue",
        "mpi_esm_mr": "red",
        "noresm1_m": "darkgreen",
        "access1_0": "lightgreen",
        "mpi_esm_lr": "orange",
        "cnrm_cm5": "purple",
    }
    # subset data
    df = df[df["spatial"] == spatial_res]

    if model == "amip":
        xticks = np.arange(1980, 2009, 1)
    elif model == "rcp":
        xticks = np.arange(1979, 2020, 1)
    else:
        raise ValueError("Unrecognized model:", model)

    fig, ax = plt.subplots(figsize=(15, 5))
    if info == "h":
        pts1 = sns.lineplot(
            x="base_time",
            y="h_base",
            data=df,
            linestyle="--",
            color="black",
            linewidth=6,
            marker="o",
            # palette=palette,
        )
        pts2 = sns.lineplot(
            x="base_time",
            y="h_cmip",
            data=df,
            hue="cmip",
            linewidth=5,
            marker="o",
            palette=palette,
        )
        ax.set_ylabel("")
        ax.set_xticklabels(xticks, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
    elif info == "mi":
        pts2 = sns.lineplot(
            x="cmip_time",
            y="mi",
            data=df,
            hue="cmip",
            linewidth=5,
            marker="o",
            palette=palette,
        )
        ax.set_xticklabels(xticks, fontsize=20)
        ax.set_ylabel("")
        ax.tick_params(axis='both', which='major', labelsize=20)
    else:
        raise ValueError("Unrecognized info measure:", info)
    plt.xticks(rotation="vertical")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        fontsize=15,
        fancybox=True,
        shadow=True,
    )
    ax.grid(True, color="gray")
    ax.set_xlabel("")
    plt.tight_layout()
    if save:
        fig.savefig(
            f"{FIG_PATH}/{model}/local/{info}_{base_model}_s{spatial_res}.png",
            transparent=True,
        )
    else:
        plt.show()


def plot_diff(df: pd.DataFrame, spatial_res: int, model: str = "amip") -> None:

    # subset data
    df = df[df["spatial"] == spatial_res]

    if model == "amip":
        xticks = np.arange(1980, 2009, 1)
    elif model == "rcp":
        xticks = np.arange(1979, 2019, 1)
    else:
        raise ValueError("Unrecognized model:", model)

    fig, ax = plt.subplots(figsize=(10, 7))

    df["h_abs_diff"] = abs(df["h_base"] - df["h_cmip"])

    pts2 = sns.lineplot(x="base_time", y="h_abs_diff", hue="cmip", data=df, linewidth=5)
    ax.set_xlabel("Time", fontsize=20)
    ax.set_xticklabels(xticks, fontsize=20)
    ax.set_ylabel("Absolute Difference, H", fontsize=20)
    plt.xticks(rotation="vertical")
    plt.show()


def plot_individual_diff(
    df: pd.DataFrame, cmip_model: str, spatial_res: int, model: str = "amip"
) -> None:

    # subset data
    df = df[df["cmip"] == cmip_model]
    df = df[df["spatial"] == spatial_res]

    if model == "amip":
        xticks = np.arange(1980, 2009, 1)
    elif model == "rcp":
        xticks = np.arange(1980, 2019, 1)
    else:
        raise ValueError("Unrecognized model:", model)

    fig, ax = plt.subplots(figsize=(10, 7))

    df["h_abs_diff"] = abs(df["h_base"] - df["h_cmip"])

    pts2 = sns.lineplot(x="base_time", y="h_abs_diff", data=df, linewidth=5)
    plt.xticks(rotation="vertical")
    ax.set_xticklabels(xticks, fontsize=20)
    plt.show()
