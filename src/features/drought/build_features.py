import sys, os

sys.path.insert(0, "/home/emmanuel/code/py_esdc")

from functools import reduce

from esdc.shape import ShapeFileExtract, rasterize
from esdc.preprocessing import calculate_monthly_mean
from esdc.transform import DensityCubes

import xarray as xr
import pandas as pd
import shapely
from typing import Union, List, Tuple


SHAPEFILES = "/media/disk/databases/SHAPEFILES/shape_files_us_states/"
EMDATSHAPE = "/media/disk/databases/SMADI/EMDAT_validation/"


def get_cali_geometry():

    # initialize shapefile extractor
    shapefiles_clf = ShapeFileExtract()

    # extract shapefiles
    shapefiles_clf.import_shape_files(SHAPEFILES)

    # extract california
    query = "name"
    subqueries = ["California"]

    # get geometries
    cali_geoms = shapefiles_clf.extract_polygons(query=query, subqueries=subqueries)

    return cali_geoms


def mask_datacube(
    ds: Union[xr.DataArray, xr.Dataset],
    geometry: shapely.geometry.multipolygon.MultiPolygon,
) -> Union[xr.Dataset, xr.DataArray]:

    # mask dataset
    ds["cali_mask"] = rasterize(geometry, ds)

    ds = ds.where(ds["cali_mask"] == 1, drop=True)

    return ds


def smooth_vod_signal(
    ds: Union[xr.DataArray, xr.Dataset], window_length=2, center=True
) -> Union[xr.DataArray, xr.Dataset]:
    ds["VOD"] = ds.VOD.rolling(time=window_length, center=center).mean()
    return ds


def remove_climatology(
    ds: Union[xr.DataArray, xr.Dataset]
) -> Union[xr.DataArray, xr.Dataset]:
    # calculate the climatology
    ds_mean = calculate_monthly_mean(ds)

    # remove climatology
    ds = ds.groupby("time.month") - ds_mean
    return ds, ds_mean


def get_cali_emdata():

    shapefiles_clf = ShapeFileExtract()

    shapefiles_clf.import_shape_files(EMDATSHAPE)

    # Extract Europe
    query = "LOCATION"
    subqueries = ["California"]

    cali_droughts = shapefiles_clf.extract_queries(query=query, subqueries=subqueries)

    return cali_droughts


def get_drought_years(
    ds: Union[xr.DataArray, xr.Dataset], years=List[str]
) -> Union[xr.Dataset, xr.DataArray]:
    ds = xr.concat([ds.sel(time=slice(iyear, iyear)) for iyear in years], dim="time")
    return ds


def get_density_cubes(
    ds: Union[xr.Dataset, xr.DataArray], spatial: int = 1, temporal: int = 12
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # initialize density cubes
    minicuber = DensityCubes(spatial_window=spatial, time_window=temporal)

    # initialize dataframes
    drought_VOD = pd.DataFrame()
    drought_LST = pd.DataFrame()
    drought_NDVI = pd.DataFrame()
    drought_SM = pd.DataFrame()

    # Group by year and get minicubes
    for iyear, igroup in ds.groupby("time.year"):
        # get minicubes for variables
        drought_VOD = drought_VOD.append(minicuber.get_minicubes(igroup.VOD))
        drought_LST = drought_LST.append(minicuber.get_minicubes(igroup.LST))
        drought_NDVI = drought_NDVI.append(minicuber.get_minicubes(igroup.NDVI))
        drought_SM = drought_SM.append(minicuber.get_minicubes(igroup.SM))
    return drought_VOD, drought_LST, drought_NDVI, drought_SM


def normalize(X: pd.DataFrame) -> pd.DataFrame:
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    return (X - X_mean) / X_std


def get_common_elements(
    X1: pd.DataFrame, X2: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = X1.index.intersection(X2.index)
    return X1.loc[idx], X2.loc[idx]


def get_common_elements_many(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:

    # get common element index
    idx = pd.concat(dfs, axis=1, join="inner").index

    # get subset elements that are common
    dfs = [df.loc[idx] for df in dfs]

    return dfs
