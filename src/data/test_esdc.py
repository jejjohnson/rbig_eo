from src.data.esdc import get_dataset
import pathlib
import pytest
from xcube.core.dsio import open_cube
import requests

ONLINE_CUBE = "https://obs.eu-de.otc.t-systems.com"
LOW_RES = "obs-esdc-v2.1/esdc-8d-0.25deg-1x720x1440-2.1.0.zarr"
CUBE_VERSION = "2.1.0"
EXT = ".zarr"
ESDC_PATH = "obs-esdc-v2.1/esdc-8d-1x720x1440-2.1.0.zarr"
INFO_EARTH = [
    "gross_primary_productivity",
    "root_moisture",
    "precipitation",
    "leaf_area_index",
    "land_surface_temperature",
]


@pytest.mark.datatest
def test_lo_esdc_online():
    path = f"{ONLINE_CUBE}/{LOW_RES}"
    esdc = open_cube(str(path))


@pytest.mark.datatest
def test_infoearth_variables():

    # load the cube from the server
    path = f"{ONLINE_CUBE}/{LOW_RES}"

    esdc = open_cube(str(path))

    for ivariable in INFO_EARTH:
        ivariable = esdc[ivariable]
