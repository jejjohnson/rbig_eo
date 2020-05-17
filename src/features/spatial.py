from collections import namedtuple
from typing import Tuple, Union

import rioxarray
import shapely
import xarray as xr
import geopandas as gpd
import shapely
from xcube.core.geom import clip_dataset_by_geometry, mask_dataset_by_geometry

RegionBox = namedtuple("RegionBox", ["name", "lonmin", "lonmax", "latmin", "latmax"])
RegionShape = namedtuple("RegionShape", ["name", "geometries"])


def get_europe() -> RegionBox:
    """As an example, I often choose Europe. This is a decent bounding box."""
    return RegionBox(name="europe", latmin=35.5, latmax=71.5, lonmin=40.0, lonmax=-18.0)


def get_northern_hemisphere() -> RegionBox:
    """As an example, I often choose Europe. This is a decent bounding box."""
    return RegionBox(name="north", latmin=0, latmax=90, lonmin=-180, lonmax=180)


def get_southern_hemisphere() -> RegionBox:
    """As an example, I often choose Europe. This is a decent bounding box."""
    return RegionBox(name="south", latmin=-90, latmax=0, lonmin=-180, lonmax=180)


def select_region(
    xr_data: Union[xr.Dataset, xr.DataArray], bbox: Union[RegionBox, RegionShape]
) -> xr.Dataset:
    """Function to spatially subset an xarray dataset from a bounding box."""
    if isinstance(xr_data, xr.DataArray):
        xr_data = xr.Dataset({f"{xr_data.name}": xr_data})
    # get bounding box
    if isinstance(bbox, RegionBox):
        bbox = shapely.geometry.box(bbox.lonmax, bbox.latmax, bbox.lonmin, bbox.latmin)
        return clip_dataset_by_geometry(xr_data, bbox)
    elif isinstance(bbox, RegionShape):
        bbox = bbox.geometries
        return mask_dataset_by_geometry(xr_data, bbox)
    else:
        raise ValueError(f"Unrecognized bbox type: {type(bbox)}")


def get_spain() -> RegionShape:

    # get shapefile
    shape_file = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # query column
    shape_file = shape_file[shape_file["name"] == "Spain"]

    # collapse shape file
    shape_file = shape_file.dissolve(by="continent").head()

    return RegionShape(name="spain", geometries=shape_file.geometry.values[0])
