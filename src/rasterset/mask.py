
import numpy as np
import numpy.ma as ma
import rasterio.features
from rasterio.windows import from_bounds

from . import window


WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90)


class MaskBase:
    def __init__(self):
        self._transform = None
        self._mask = None
        self._bounds = None
        return

    def eval(self, bounds):
        raise NotImplementedError()
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


class NullMask(MaskBase):
    def __init__(self):
        super().__init__()
        self._bounds = WORLD_BOUNDS
        return

    def eval(self, bounds):
        win = from_bounds(*bounds, self.transform)
        self._mask = np.full(window.shape(win), 0, dtype='uint8')
        return


class ShapesMask(MaskBase):
    def __init__(self, shapes, all_touched):
        super().__init__()
        self._all_touched = all_touched
        self._bounds = getattr(shapes, 'bounds', None)
        if not self._bounds:
            self._bounds = window.union(shapes)
        self._shapes = [feature["geometry"] for feature in shapes]
        return

    def eval(self, bounds):
        assert self.transform is not None
        win = from_bounds(*bounds, self.transform)
        self._mask = rasterio.features.geometry_mask(
            self._shapes,
            transform=self.transform,
            invert=False,
            out_shape=window.shape(win),
            all_touched=self._all_touched,
        )
        return


class RasterMask(MaskBase):
    def __init__(self, ds, mask_val=1.0):
        super().__init__()
        self._bounds = ds.bounds
        self._ds = ds
        self._mask_val = mask_val
        return

    def eval(self, bounds):
        win = self._ds.window(*bounds)
        data = self._ds.read(1, masked=True, window=win)
        self._mask = ma.where(data == self._mask_val, True, False).filled(True)
        return


def mask_maker(shapes=None, all_touched=None, ds=None, mask_val=None):
    if shapes is not None:
        return ShapesMask(shapes, all_touched)
    if ds is not None:
        return RasterMask(ds, mask_val)
    return NullMask()
