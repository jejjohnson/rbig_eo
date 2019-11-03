import argparse

# drought tools
from src.data.drought.loader import DataLoader
from src.features.drought.build_features import (
    get_cali_geometry,
    mask_datacube,
    smooth_vod_signal,
    remove_climatology,
    get_cali_emdata,
    get_drought_years,
    get_density_cubes,
    get_common_elements_many,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.train_models import run_rbig_models
from tqdm import tqdm

DATA_PATH = "/home/emmanuel/projects/2020_rbig_rs/data/drought/results/"


def main(args):

    # Load data
    drought_cube = DataLoader().load_data(args.region, args.sampling)

    # get cali geometry
    cali_geoms = get_cali_geometry()

    # subset datacube with cali
    drought_cube = mask_datacube(drought_cube, cali_geoms)

    # do interpolation
    drought_cube = drought_cube.interpolate_na(dim="time", method=args.interp_method)

    # Remove climatology
    drought_cube, _ = remove_climatology(drought_cube)

    # drought_years
    drought_years = [
        ("2010", False),
        ("2011", False),
        ("2012", True),
        ("2013", False),
        ("2014", True),
        ("2015", True),
    ]

    time_steps = range(1, 12)
    spatial = 1
    results_df_single = pd.DataFrame()

    # group datacube by years
    with tqdm(drought_cube.groupby("time.year")) as years_bar:
        for iyear, icube in years_bar:

            # Loop through time steps
            for itime_step in time_steps:

                # extract density cubes
                vod_df, lst_df, ndvi_df, sm_df = get_density_cubes(
                    icube, spatial, itime_step
                )

                # get common elements
                dfs = get_common_elements_many([vod_df, lst_df, ndvi_df, sm_df])
                vod_df, lst_df, ndvi_df, sm_df = dfs[0], dfs[1], dfs[2], dfs[3]

                variables = {"VOD": vod_df, "NDVI": ndvi_df, "SM": sm_df, "LST": lst_df}

                # do calculations for H, TC
                for iname, idata in variables.items():

                    # normalize data
                    X_norm = StandardScaler().fit_transform(idata)

                    # entropy, total correlation
                    tc, h, t_ = run_rbig_models(X_norm, measure="t", random_state=123)

                    # get H and TC
                    results_df_single = results_df_single.append(
                        {
                            "samples": X_norm.shape[0],
                            "dimensions": X_norm.shape[1],
                            "temporal": itime_step,
                            "variable": iname,
                            "tc": tc,
                            "h": h,
                            "time": t_,
                        },
                        ignore_index=True,
                    )

                    results_df_single.to_csv(DATA_PATH + args.save_name)
                    postfix = dict(
                        Year=f"{iyear}", Dims=f"{itime_step}", Variable=f"{iname}"
                    )
                    years_bar.set_postfix(postfix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Drought Experiment.")

    # DataCube Arguments
    parser.add_argument(
        "--region", default="conus", type=str, help="The region for the drought events."
    )
    parser.add_argument(
        "--sampling",
        default="14D",
        type=str,
        help="The sampling scheme for drought events.",
    )

    # PreProcessing Arguments
    parser.add_argument(
        "--interp_method", default="linear", type=str, help="Interpolation method."
    )

    # Climatology Arguments
    parser.add_argument(
        "--climatology_window",
        default=2,
        type=int,
        help="Window length for climatology.",
    )

    # logistics
    parser.add_argument(
        "--save_name",
        default="exp_trial_v1.csv",
        type=str,
        help="Save Name for data results.",
    )
    main(parser.parse_args())
