#%%

import shapely
import shapely.vectorized
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import pandas as pd


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


#%%


x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)

xx, yy = np.meshgrid(x, y)

fig, ax = plt.subplots()
ax.scatter(xx, yy)
ax.set(xlim=[-2, 2], ylim=[-2, 2])
plt.show()
#%%
# create geometry
# geometry = [Point(bounds) for bounds in zip(x_bounds, y_bounds)]
vertices = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]

lines = np.vstack((vertices))

#%%
# mpath = Path(vertices, codes)
mpath = Path(lines)

fig, ax = plt.subplots()
patch = patches.PathPatch(mpath, facecolor="orange", lw=2)
ax.add_patch(patch)
ax.set(xlim=[-2, 2], ylim=[-2, 2])
plt.show()

#%%

# check for points inside
points = np.array((xx.flatten(), yy.flatten())).T

mask = mpath.contains_points(points).reshape(xx.shape)

fig, ax = plt.subplots()
ax.scatter(xx[mask], yy[mask])
ax.set(xlim=[-2, 2], ylim=[-2, 2])
plt.show()

# %%

# create geometry
geometry = [Point(xy) for xy in vertices]
# create array of data
df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()})

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

#%% Cut Geometry

from shapely.geometry import Polygon

geom = Polygon(vertices)

# mask
mask_ = shapely.vectorized.contains(geom, xx, yy)
fig, ax = plt.subplots()
ax.scatter(xx[mask_], yy[mask_])
ax.set(xlim=[-2, 2], ylim=[-2, 2])
plt.show()


# %%
