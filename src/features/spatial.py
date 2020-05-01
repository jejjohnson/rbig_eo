from collections import namedtuple
from typing import Tuple, Union

import rioxarray
import shapely
import xarray as xr
from xcube.core.geom import clip_dataset_by_geometry

Region = namedtuple("Region", ["name", "lonmin", "lonmax", "latmin", "latmax"])


def get_europe() -> Region:
    """As an example, I often choose Europe. This is a decent bounding box."""
    return Region(name="europe", latmax=35.5, latmin=71.5, lonmax=40.0, lonmin=-18.0)


def select_region(xr_data: xr.Dataset, bbox: Region) -> xr.Dataset:
    """Function to spatially subset an xarray dataset from a bounding box."""
    # get bounding box
    bbox = shapely.geometry.box(bbox.lonmin, bbox.latmin, bbox.lonmax, bbox.latmax)
    # subset datacube
    return clip_dataset_by_geometry(xr_data, bbox)
