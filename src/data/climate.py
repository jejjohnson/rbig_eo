import pandas as pd
from typing import List, Optional


class LoadResults:
    def __init__(self, filenames: List[str]):
        self.filenames = filenames

    def load_dataframes(self, filenames: Optional[List[str]] = None):
        if filenames is None:
            filenames = self.filenames

        results = pd.DataFrame()
        for ifile in filenames:

            # append results
            new = pd.read_csv(f"{ifile}", index_col=[0])
            #             print(new.head())
            results = results.append(new, ignore_index=True)

        #         results = results.drop()
        return results
