
from functools import reduce
import numpy as np
from rasterio.coords import BoundingBox, disjoint_bounds


def intersection(*bounds):
    stacked = np.dstack(bounds)
    return BoundingBox(stacked[0, 0].max(), stacked[0, 1].max(),
                       stacked[0, 2].min(), stacked[0, 3].min())


def union(*bounds):
    stacked = np.dstack(bounds)
    return BoundingBox(stacked[0, 0].min(), stacked[0, 1].min(),
                       stacked[0, 2].max(), stacked[0, 3].max())


def disjoint(bounds):
    return reduce(disjoint_bounds, bounds,
                  (-180.0, -90.0, 180.0, 90.0))


def shape(win):
    return (win.height, win.width)


def round(win):
    return win.round_offsets('floor').round_lengths('ceil')
