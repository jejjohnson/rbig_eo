import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_individual(df: pd.DataFrame, cmip_model: str, spatial_res: int) -> None:

    plt.style.use(["seaborn-poster"])

    # subset data
    df = df[df["cmip"] == cmip_model]
    df = df[df["spatial"] == spatial_res]

    fig, ax = plt.subplots(figsize=(10, 7))

    pts1 = sns.lineplot(x="base_time", y="h_base", data=df)
    pts2 = sns.lineplot(x="base_time", y="h_cmip", data=df)

    plt.show()


def plot_diff(df: pd.DataFrame, spatial_res: int) -> None:

    plt.style.use(["seaborn-poster"])

    # subset data
    df = df[df["spatial"] == spatial_res]

    fig, ax = plt.subplots(figsize=(10, 7))

    df["h_abs_diff"] = abs(df["h_base"] - df["h_cmip"])

    pts2 = sns.lineplot(x="base_time", y="h_abs_diff", hue="cmip", data=df)
    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute Difference, H")

    plt.show()


def plot_individual_diff(df: pd.DataFrame, cmip_model: str, spatial_res: int) -> None:

    plt.style.use(["seaborn-poster"])

    # subset data
    df = df[df["cmip"] == cmip_model]
    df = df[df["spatial"] == spatial_res]

    fig, ax = plt.subplots(figsize=(10, 7))

    df["h_abs_diff"] = abs(df["h_base"] - df["h_cmip"])

    pts2 = sns.lineplot(x="base_time", y="h_abs_diff", data=df)

    plt.show()
