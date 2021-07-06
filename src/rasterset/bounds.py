from functools import reduce
import math

import numpy as np
from rasterio.coords import BoundingBox, disjoint_bounds


WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90)


def intersection(*bounds):
    stacked = np.dstack(bounds)
    return BoundingBox(stacked[0, 0].max(), stacked[0, 1].max(),
                       stacked[0, 2].min(), stacked[0, 3].min())


def union(*bounds):
    stacked = np.dstack(bounds)
    return BoundingBox(stacked[0, 0].min(), stacked[0, 1].min(),
                       stacked[0, 2].max(), stacked[0, 3].max())


def disjoint(*bounds):
    return reduce(disjoint_bounds, bounds,
                  (-180.0, -90.0, 180.0, 90.0))


def round(left, bottom, top, right, res):
    def round_down(x, a):
        return math.floor(x / a) * a

    def round_up(x, a):
        return math.ceil(x / a) * a

    def round_to_inf(x, a):
        if a * x < 0:
            return round_down(x, a)
        return round_up(x, a)
    return BoundingBox(round_to_inf(left, res[0]),
                       round_to_inf(bottom, res[1]),
                       round_to_inf(top, res[0]),
                       round_to_inf(right, res[1])
                       )
