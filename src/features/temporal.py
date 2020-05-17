from collections import namedtuple
from typing import Tuple, Union

import xarray as xr

TimePeriod = namedtuple("TimePeriod", ["name", "start", "end"])


def get_smoke_test_time() -> TimePeriod:
    return TimePeriod(name="test_201007", start="June-2010", end="August-2010")


def get_summer_time() -> TimePeriod:
    return TimePeriod(name="summer_2010", start="June-2010", end="August-2010")


def get_winter_time() -> TimePeriod:
    return TimePeriod(name="winter_2010", start="Dec-2009", end="Feb-2010")


def get_fall_time() -> TimePeriod:
    return TimePeriod(name="fall_2010", start="Sept-2010", end="Nov-2010")


def get_spring_time() -> TimePeriod:
    return TimePeriod(name="spring_2010", start="March-2010", end="May-2010")


def select_period(xr_data: xr.DataArray, period: TimePeriod) -> xr.DataArray:
    """Function to temporally subset an xarray dataset from a tuple of
    start date and end date
    """
    return xr_data.sel(time=slice(period.start, period.end))


def calculate_monthly_mean(
    xr_obj: Union[xr.DataArray, xr.Dataset]
) -> Union[xr.DataArray, xr.Dataset]:
    dims = [dim for dim in xr_obj.coords.keys()]
    msg = f"Time must be in dataset dimensions. Currently: {dims}"
    assert "time" in dims, msg
    return xr_obj.groupby("time.month").mean(dim="time")


def remove_climatology(
    ds: Union[xr.DataArray, xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    # calculate the climatology
    ds_mean = calculate_monthly_mean(ds)

    # remove climatology
    ds = ds.groupby("time.month") - ds_mean
    return ds, ds_mean
