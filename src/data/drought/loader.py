import sys

sys.path.insert(0, "/home/emmanuel/code/py_esdc")

# standard packages
import xarray as xr
import pandas as pd
import numpy as np

# esdc tools
from esdc.subset import select_pixel
from esdc.shape import ShapeFileExtract, rasterize
from esdc.transform import DensityCubes

DATA_PATH = "/media/disk/databases/DROUGHT/conus/"


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, region="conus", sampling="14D"):

        # load cube
        drought_cube = xr.open_dataset(f"{DATA_PATH}AD_{region}_{sampling}.nc")

        return drought_cube
