import sys

sys.path.insert(0, f"/home/emmanuel/code/py_esdc")
from esdc.transform import DensityCubes
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from sklearn import preprocessing

# ESDC tools


xr_dat = Union[xr.Dataset, xr.DataArray]


def get_time_overlap(
    ds1: Union[xr.Dataset, xr.DataArray], ds2: Union[xr.Dataset, xr.DataArray]
) -> Tuple[Union[xr.Dataset, xr.DataArray], Union[xr.Dataset, xr.DataArray]]:

    # check time_coordinates
    ds1 = check_time_coords(ds1)
    ds2 = check_time_coords(ds2)

    # get min,max time slices
    start_time = max(ds1.time.data[0], ds2.time.data[0])
    end_time = min(ds1.time.data[-1], ds2.time.data[-1])

    # subset arrays
    ds1 = ds1.sel(time=slice(start_time, end_time))
    ds2 = ds2.sel(time=slice(start_time, end_time))
    return ds1, ds2


def check_time_coords(
    ds: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:

    if isinstance(ds.indexes["time"], xr.coding.cftimeindex.CFTimeIndex):
        time_coords = ds.indexes["time"].to_datetimeindex(True)
        ds["time"] = time_coords
    return ds


def regrid_2_lower_res(ds1: xr_dat, ds2: xr_dat) -> Tuple[xr_dat, xr_dat]:

    ds1_coords = len(ds1.lat) + len(ds1.lon)
    ds2_coords = len(ds2.lat) + len(ds2.lon)

    if ds1_coords >= ds2_coords:
        model_id = ds1.attrs["model_id"]
        ds1 = regrid_data(ds2, ds1)
        ds1.attrs["model_id"] = model_id
    else:
        model_id = ds2.attrs["model_id"]
        ds2 = regrid_data(ds1, ds2)
        ds2.attrs["model_id"] = model_id

    return ds1, ds2


def regrid_data(
    reference_ds: Union[xr.Dataset, xr.DataArray],
    ds: Union[xr.Dataset, xr.DataArray],
    method: Optional[str] = "nearest_s2d",
    clean: bool = True,
) -> xr.Dataset:
    """A light wrapper over the xEMSF package used to regrid datasets. Will
    regrid the new dataset under the reference dataset.

    Parameters
    ----------
    reference_ds : xr.Dataset, xr.DataArray
        The reference dataset with original grid information

    ds : xr.Dataset, xr.DataArray
        The dataset to be regridded
    
    method : str, default='nearest_s2d'
        The regridding method

    clean : bool, default=True
        Removes weight file afterwards
    Returns
    -------
    ds : xr.Dataset, xr.DataArray
        The regridded dataset.

    Examples
    --------
    >> data_new = regrid_data(
        reference_ds, ds, method='nearest_s2d'
    )
    """

    # Assert reference dims are the same
    assert ("lat" in reference_ds.dims) & (
        "lon" in reference_ds.dims
    ), f"Need (lat,lon) in reference_ds dims. Currently: {reference_ds.dims}"
    assert ("lat" in ds.dims) & (
        "lon" in ds.dims
    ), f"Need (lat,lon) in ds dims. Currently: {ds.dims}"

    # create grid we want to convert to
    ds_out = xr.Dataset(
        {"lat": (["lat"], reference_ds.lat), "lon": (["lon"], reference_ds.lon)}
    )
    # create regridder object
    regridder = xe.Regridder(ds, ds_out, method, reuse_weights=True)

    # case of dataarray
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = regridder(ds)
    elif isinstance(ds, xr.core.dataset.Dataset):
        variables = [i for i in ds.var().variables]

        if len(variables) == 1:
            ds = regridder(ds)
        else:

            output_dict = {}

            # loop over each variable and append
            for ivar in variables:
                da = ds[ivar]
                da = regridder(da)
                output_dict[ivar] = da

            # Rebuild
            ds = xr.Dataset(output_dict)
    else:
        raise ValueError(f"Unrecognized type of object.")
    if clean:
        regridder.clean_weight_file()
    return ds


def get_spatial_cubes(da: xr.DataArray, spatial_window: int) -> pd.DataFrame:
    minicuber = DensityCubes(spatial_window=spatial_window, time_window=1)

    df = minicuber.get_minicubes(da)
    return df


def normalize_data(df: pd.DataFrame) -> np.ndarray:
    return preprocessing.StandardScaler().fit_transform(df.values)


def add_empty_time_coord(da: xr.DataArray) -> xr.DataArray:
    da = da.expand_dims({"time": 1})
    return da
