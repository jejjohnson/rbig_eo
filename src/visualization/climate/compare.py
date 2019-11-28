import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_individual(df: pd.DataFrame, cmip_model: str, spatial_res: int) -> None:

    plt.style.use(["seaborn-poster"])

    # subset data
    df = df[df["cmip"] == cmip_model]
    df = df[df["spatial"] == spatial_res]

    fig, ax = plt.subplots(figsize=(10, 7))

    pts1 = sns.lineplot(x="base_time", y="mi", hue="cmip", data=df)

    plt.show()


def plot_all(df: pd.DataFrame, spatial_res: int, stat: str) -> None:

    plt.style.use(["seaborn-poster"])

    # subset data
    df = df[df["spatial"] == spatial_res]

    fig, ax = plt.subplots(figsize=(10, 7))

    pts1 = sns.lineplot(x="base_time", y=stat, hue="cmip", data=df)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mutual Information")

    plt.show()
