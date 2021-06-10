
import numpy as np
import numpy.ma as ma
import rasterio.features
from rasterio.windows import from_bounds

from . import window


WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90)


class Mask:
    def __init__(self, shapes, mask_ds, maskval, all_touched):
        self._ds = None
        self._shapes = None
        self._maskval = maskval
        self._all_touched = all_touched
        self._transform = None
        self._mask = None
        if shapes:
            self._bounds = getattr(shapes, 'bounds', None)
            if not self._bounds:
                self._bounds = window.union(shapes)
            self._shapes = [feature["geometry"] for feature in shapes]
            self._type = 'shape'
            return
        if mask_ds:
            self._bounds = mask_ds.bounds
            self._ds = mask_ds
            self._type = 'raster'
            return
        self._type = None
        self._bounds = WORLD_BOUNDS
        return

    def read(self, window):
        assert self._ds is not None
        data = self.mask_ds.read(1, masked=True, window=window)
        return ma.where(data == self._maskval, True, False).filled(True)

    def rasterize(self, window):
        assert self.transform is not None
        return rasterio.features.geometry_mask(
            self._shapes,
            transform=self.transform,
            invert=False,
            out_shape=window.shape(window),
            all_touched=self._all_touched,
        )

    def eval(self, bounds):
        if self._mask is not None:
            return self.mask
        if self._type == 'raster':
            self._mask = self.read(bounds)
            return
        win = from_bounds(*bounds, self.transform)
        if self._type is None:
            self._mask = np.full(window.shape(win), 0, dtype='uint8')
            return
        if self._type == 'shapes':
            self._mask = self.rasterize(win)
            return
        assert RuntimeError(f"Unknown window type {self._type}")
        return

    @property
    def bounds(self):
        return self._bounds

    @property
    def mask(self):
        return self._mask

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform
        return
