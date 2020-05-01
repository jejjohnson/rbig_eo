import numpy as np
import pandas as pd
import skimage
import xarray as xr

LEVELS = ["time", "lat", "lon"]


class DensityCubes:
    """A class that extracts subsets of cubes from the xarray datacube. This cube is designed
    to work with xr.DataArray structures with 2 spatial dimensions and 1 temporal (e.g. 
    lat-lon-time). This is useful for density estimation and feature inclusion when we want to have 
    more features/dimensions that stem from the spatial and temporal dimensions. Uses 
    numpy strides to reduce the amount of memory overhead and is relatively quick.
    
    Parameters
    ----------
    spatial_window : int, default=3
        The number of dimensions for the spatial windows (x2) for the cube.
        Note: if the time window is even then we take the ceil 

    time_window : int, default = 3
        The number of dimensions for the temporal window (x1) for the cube.

    spatial_step : int, default = 1
        The amount of steps in the spatial direction for the spatial windows.
    
    time_step : int, default = 1
        The amount of steps in the temporal direction for the temporal windows.
    
    temporal_point : str, default = "end"
        The point where we extract the time dimension
        * 'begin' - first time step
        * 'mid' - mid time step
        * 'end' - end time step
        Note: if the time window is even then we take the ceil of the middle

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 15 - June - 2019

    Example
    -------
    >> spatial_resolution = 3
    >> temporal_resolution = 3
    >> minicuber = DensityCubes(
        spatial_window=spatial_resolution,
        time_window=temporal_resolution
    )
    >> minicube_df = minicuber.get_minicubes(datacube)
    """

    def __init__(
        self,
        spatial_window: int = 3,
        time_window: int = 3,
        spatial_step: int = 1,
        time_step: int = 1,
        temporal_point: str = "end",
    ) -> None:
        self.spatial_window = spatial_window
        self.time_window = time_window
        self.total_dims = spatial_window ** 2 * time_window
        self.temporal_point = temporal_point
        self.spatial_step = spatial_step
        self.time_step = time_step

        # spatial pixel
        if spatial_window % 2 == 0:
            self.spatial_pixel = int(np.ceil((spatial_window + 1) / 2 - 1))
        else:
            self.spatial_pixel = int((spatial_window + 1) / 2 - 1)

        # temporal pixel
        if temporal_point == "end":
            self.time_pixel = -1
        elif temporal_point == "mid":
            if time_window % 2 == 0:
                self.time_pixel = int((time_window + 1) / 2 - 1)
            else:
                self.time_pixel = int(np.ceil((time_window + 1) / 2 - 1))
        elif temporal_point == "beg":
            self.time_pixel = 0
        else:
            raise ValueError(f"Unrecognized time_pixel: '{self.time_pixel}'")

        self.window_shape = (time_window, spatial_window, spatial_window)

    def get_minicubes(
        self, datacube: xr.DataArray, variable: str = "var", dropna: bool = True
    ) -> pd.DataFrame:
        """Extracts minicubes
        
        Parameters
        ----------
        datacube : xr.DataArray
            The xr.DataArray to be minicube(-ed)
        
        Returns
        -------
        minicubes : pd.DataFrame
            A multindex dataframe (time-lat-lon) with all of the 
            samples as rows and features (spatial-temporal dimensions)
            as columns.
        """

        # check names
        req_dims = ["time", "lat", "lon"]
        dims = [dim for dim in datacube.coords.keys()]

        assert all([i in dims for i in req_dims])

        # transpose
        datacube = datacube.transpose("time", "lat", "lon")

        # View as Minicubes
        minicubes = skimage.util.view_as_windows(
            datacube.values,
            window_shape=self.window_shape,
            step=(self.spatial_step, self.spatial_step, self.time_step),
        )

        # lon, lat, time minicubees
        loncubes = skimage.util.view_as_windows(
            datacube.lon.values, window_shape=self.spatial_window
        )

        latcubes = skimage.util.view_as_windows(
            datacube.lat.values, window_shape=self.spatial_window
        )

        timecubes = skimage.util.view_as_windows(
            datacube.time.values, window_shape=self.time_window
        )
        # Lat, Lon, Time Coordinate for cube
        latcubes = latcubes[:, self.spatial_pixel]
        loncubes = loncubes[:, self.spatial_pixel]
        timecubes = timecubes[:, self.time_pixel]

        # check that the dimensions match the
        assert timecubes.shape[0] == minicubes.shape[0]
        assert latcubes.shape[0] == minicubes.shape[1]
        assert loncubes.shape[0] == minicubes.shape[2]

        # Create Labels DataFrame
        x_df = pd.DataFrame(
            data=minicubes.reshape(-1, self.total_dims),
            index=pd.MultiIndex.from_product(
                [timecubes, latcubes, loncubes], names=["time", "lat", "lon"]
            ),
            columns=[f"{variable}_x{i}" for i in range(self.total_dims)],
        )
        if dropna:
            # Drop any rows that have NANs
            x_df = x_df.dropna(axis=0)

        return x_df

    def _check_coordinates(self, datacube: xr.DataArray) -> None:

        # get names
        names = datacube.coords.dims

        assert ("time", "lat", "lon") == names


def get_density_cubes(data: xr.DataArray, spatial: int, temporal: int) -> pd.DataFrame:
    """Wrapper Function to get density cubes from a dataarray"""
    return (
        DensityCubes(spatial_window=spatial, time_window=temporal)
        .get_minicubes(data)
        .reorder_levels(LEVELS)
    )


def buff(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Implements a sliding window taking the last {window} observations
    to generate an N x window sized cube.
    
    Parameters
    ----------
    df : pd.DataFrame (n_samples)
        The timeseries in equations
    
    window : int (default = 5)
        Size of the window length

    Returns
    -------
    df : pd.DataFrame (n_samples - window x window)
        The timeseries with each feature as a previous view

    Example
    -------
    >> 
    """
    time_stamps = df.index
    T = len(time_stamps)

    # Get DataFrame
    data = pd.concat([df.shift(i) for i in range(window)], axis=1)

    # Drop last few of size w
    data = data.drop(df.index[-window + 1 :])

    index = time_stamps[window - 1 :]
    columns = [f"feat{i+1}" for i in range(window)]

    assert data.shape[1] == len(columns)
    data.index = index
    data.columns = columns[::1]
    return data
