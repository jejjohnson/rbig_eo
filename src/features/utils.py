import pandas as pd


def move_variables(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    #     cond1 = df['variable1'] == variable
    cond = df["variable2"] == variable
    df.loc[cond, ["variable2", "variable1"]] = df.loc[
        cond, ["variable1", "variable2"], ["rbig_H_x", "rbig_H_y"]
    ].values

    return df
