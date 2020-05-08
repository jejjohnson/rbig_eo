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
    return RegionBox(name="europe", latmax=35.5, latmin=71.5, lonmax=40.0, lonmin=-18.0)


def select_region(
    xr_data: xr.Dataset, bbox: Union[RegionBox, RegionShape]
) -> xr.Dataset:
    """Function to spatially subset an xarray dataset from a bounding box."""
    # get bounding box
    if isinstance(bbox, RegionBox):
        bbox = shapely.geometry.box(bbox.lonmin, bbox.latmin, bbox.lonmax, bbox.latmax)
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
