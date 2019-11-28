import pandas as pd
from typing import Optional, Tuple, List

import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
plt.style.use(["seaborn-poster", "fivethirtyeight"])

DATA_PATH = f"/home/emmanuel/projects/2020_rbig_rs/data/climate/raw/"
RESULTS_PATH = f"/home/emmanuel/projects/2020_rbig_rs/data/climate/results/"
FIG_PATH = f"/home/emmanuel/projects/2020_rbig_rs/reports/figures/climate/"


class PlotResults:
    def __init__(self, dataframe: pd.DataFrame):
        self.fig_path = "/home/emmanuel/projects/2020_rbig_rs/reports/figures/climate/"

        # load dataframe
        self.results = dataframe

    def plot_entropy(self):

        # plot the entropy
        fig, ax = plt.subplots(figsize=(15, 5))

        sns.lineplot(
            ax=ax, x="year", y="h", hue="model", data=self.results, linewidth=4
        )
        # sns.lineplot(ax=ax, x='nm_features', y='I', data=ndvi_df, label=label_main, linewidth=4,)
        ax.set_xlabel("Years", fontsize=20)
        ax.set_ylabel("Entropy", fontsize=20)
        # ax.set_ylim([34, 38])
        plt.legend(fontsize=20)
        plt.show()
        #         fig.savefig(f"{self.fig_path}{self.variable}_h.png")

        return fig, ax

    def plot_total_correlation(self):
        # plot the entropy
        fig, ax = plt.subplots(figsize=(15, 5))

        sns.lineplot(
            ax=ax, x="year", y="tc", hue="model", data=self.results, linewidth=4
        )
        # sns.lineplot(ax=ax, x='nm_features', y='I', data=ndvi_df, label=label_main, linewidth=4,)
        ax.set_xlabel("Years", fontsize=20)
        ax.set_ylabel("Total Correlation", fontsize=20)
        plt.legend(fontsize=20)
        plt.show()
        #         fig.savefig(f"{self.fig_path}{self.variable}_tc.png")

        return fig, ax

    def plot_mutual_information(
        self, omit_models: Optional[Tuple[str, List[str]]] = None
    ):

        # omit models
        if omit_models is not None:
            results = self.results[~self.results[omit_models[0]].isin(omit_models[1])]
        else:
            results = self.results
        # plot the mutual information
        fig, ax = plt.subplots(figsize=(15, 5))

        sns.lineplot(ax=ax, x="year", y="mi", hue="model", data=results, linewidth=4)
        # sns.lineplot(ax=ax, x='nm_features', y='I', data=ndvi_df, label=label_main, linewidth=4,)
        ax.set_xlabel("Years", fontsize=20)
        ax.set_ylabel("Mutual Information", fontsize=20)
        plt.legend(fontsize=20)
        plt.show()
        #         fig.savefig(f"{self.fig_path}{self.variable}_mi.png")

        return fig, ax
