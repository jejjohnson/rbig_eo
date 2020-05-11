import pathlib
import xarray as xr
from typing import List, Optional

# Datapath
DATA_PATH = pathlib.Path("/media/disk/databases/ESDC/")
# get filename
LOW_SPATIAL = DATA_PATH.joinpath("esdc-8d-0.25deg-1x720x1440-2.0.0.zarr")
HIGH_SPATIAL = DATA_PATH.joinpath("esdc-8d-0.083deg-1x2160x4320-2.0.0.zarr")


def get_dataset(
    variable: Optional[List[str]] = None, spa_resolution: str = "low"
) -> xr.Dataset:
    if spa_resolution == "high":
        filename = HIGH_SPATIAL
    elif spa_resolution == "low":
        filename = LOW_SPATIAL
    else:
        raise ValueError(f"Unrecognized spatial resolution: {spa_resolution}")

    if variable is not None:
        return xr.open_zarr(str(filename))[variable]
    else:
        return xr.open_zarr(str(filename))
