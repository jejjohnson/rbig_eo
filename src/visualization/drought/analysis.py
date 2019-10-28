import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

plt.style.use(["fivethirtyeight", "seaborn-poster"])


def plot_mean_time(da: xr.DataArray) -> None:
    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes(projection=ccrs.PlateCarree())

    da.mean(dim="time").plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        robust=True,
        cbar_kwargs={"shrink": 0.5},
    )

    ax.set_title("Land Surface Temperature")
    ax.coastlines()
    plt.show()
