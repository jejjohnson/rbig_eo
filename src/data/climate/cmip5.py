import cdsapi
import xarray as xr
from typing import Union
from src.data.climate.amip import DataLoader as DLAMIP
from src.data.climate.rcp import DataLoader as DLRCP


DATA_DIR = "/home/emmanuel/projects/2020_rbig_rs/data/climate/raw/"

xr_types = Union[xr.Dataset, xr.DataArray]


def get_cmip5_model(cmip_model: str, variable: str, model: str = "amip") -> xr_types:

    if model == "amip":
        loader = DLAMIP()
        ds = loader.load_amip_data(cmip_model)[variable]
    elif model == "rcp":
        loader = DLRCP()
        ds = loader.load_rcp_data(cmip_model)[variable]
    else:
        raise ValueError("Unrecognized model:", model)
    ds.attrs["model_id"] = cmip_model
    return ds


def main():

    get_data()

    return None


if __name__ == "__main__":
    main()
