import xarray as xr

NCEP_PATH = f"/home/emmanuel/projects/2020_rbig_rs/data/climate/raw/ncep/"


def get_ncep_data():

    ncep_data = xr.open_mfdataset(f"{NCEP_PATH}*mon.mean.nc", combine="by_coords")
    ncep_data = ncep_data.rename({"mslp": "psl"})
    ncep_data.attrs["model_id"] = "ncar_ncep_doe_2"
    return ncep_data

