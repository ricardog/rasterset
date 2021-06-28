from affine import Affine
import numpy as np
from rasterio.crs import CRS
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import rowcol
from rasterio.windows import Window

from .mask import mask_maker
from . import Raster
from . import bounds
from . import window


WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90)
EPSG_4326 = CRS.from_epsg(4326)

class EvalContext(object):
    def __init__(self, rasterset, what, crop=True, bbox=None):
        self._rasterset = rasterset
        self._what = what
        self._crop = crop
        self._crs = None
        self._res = None
        self._msgs = True
        self._mask = None
        self._bounds = None
        self._transform = None
        self._nodata = -9999
        self._window = None
        self._shape = None
        self._block_shape = None
        self._needed = sorted(rasterset.find_needed(what), key=lambda x: x.lower())
        self._sources = tuple(filter(lambda c: isinstance(c, Raster),
                                     [rasterset[x] for x in self._needed]))
        assert self._sources, "No rasters of know size in input set"

        # Check all rasters have the same resolution and CRS.
        # TODO: scale/reproject rasters as needed.
        self._res, self._crs = self.check_rasters()
        assert self._crs is not None

        # Compute bounds as intersection of all raster bounds.
        self._bounds = bounds.intersection(bbox or WORLD_BOUNDS,
                                           *(s.bounds for s in self.sources))

        self._mask = mask_maker(rasterset.shapes, rasterset.all_touched,
                                rasterset.mask, rasterset.maskval)
        # Crop output to bounds of mask.
        if crop:
            self._bounds = bounds.intersection(self.bounds, self.mask.bounds)

        # Set the affine transform and calculate mask.
        self._transform = (Affine.translation(self.bounds.left,
                                              self.bounds.top) *
                           Affine.scale(self.res[0], self.res[1]) *
                           Affine.identity())
        self.mask.transform = self.transform
        self.mask.eval(self.bounds)
        return

    def check_rasters(self):
        def check_alignment(src):
            x_off = src.transform[2] % src.transform[0]
            y_off = src.transform[5] % abs(src.transform[4])
            xx = x_off < 1e-10 or (abs(x_off - src.transform[0]) < 1e-10)
            yy = y_off < 1e-10 or (abs(y_off - abs(src.transform[4])) < 1e-10)
            return xx and yy

        first = self.sources[0]
        first_res = first.res
        first_crs = first.crs
        # Verify all rasters either have same CRS or don't have a CRS
        for source in self.sources:
            if (first_crs is None or first_crs.to_string() == "") and (
                source.crs is not None and source.crs.to_string() != ""
            ):
                first_crs = source.crs
            elif source.crs is None or source.crs.to_string() == "":
                pass
            elif first_crs != source.crs:
                raise RuntimeError(
                    "%s: crs mismatch (%s != %s)" % (source.name, first_crs, source.crs)
                )
            if not np.allclose(first_res, source.res):
                # FIXME: source no longer has a name attribute
                raise RuntimeError(
                    "%s: resolution mismatch (%s != %s)"
                    % (source, first_res, source.res)
                )
        for col in self.sources:
            if not check_alignment(col):
                print("WARNING: raster %s has unaligned pixels" % col)

        return first_res, first_crs

    def block_windows(self):
        y_inc, x_inc = self.block_shape
        for j in range(0, self.height, y_inc):
            j2 = min(j + y_inc, self.height)
            for i in range(0, self.width, x_inc):
                i2 = min(i + x_inc, self.width)
                yield Window(col_off=i, row_off=j,
                             width=i2 - i, height=j2 - j)

    def need(self, what):
        return what in self._needed

    def meta(self, args={}):
        meta = DefaultGTiffProfile(count=1,
                                   dtype=np.float32,
                                   predictor=3,
                                   crs=self.crs,
                                   nodata=self._nodata,
                                   transform=self.transform,
                                   width=self.width,
                                   height=self.height,
                                   sparse_ok="YES",
                                   )
        meta.update(args)
        return meta

    @property
    def sources(self):
        return self._sources

    @property
    def bounds(self):
        return self._bounds

    @property
    def transform(self):
        return self._transform

    @property
    def res(self):
        return self._res

    @property
    def crs(self):
        return self._crs

    @property
    def window(self):
        if self._window is not None:
            return self._window
        (left, bottom, right, top) = self.bounds
        rows, cols = rowcol(
            self.transform,
            [left, right, right, left],
            [top, top, bottom, bottom],
            op=float,
            precision=1e-5,
        )
        row_start, row_stop = min(rows), max(rows)
        col_start, col_stop = min(cols), max(cols)

        self._window = window.round(Window(
            col_off=col_start,
            row_off=row_start,
            width=max(col_stop - col_start, 0.0),
            height=max(row_stop - row_start, 0.0),
        ))
        return self._window

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        self._shape = window.shape(self.window)
        return self._shape

    @property
    def block_shape(self):
        if self._block_shape:
            return self.block_shape
        block_shapes = [s.block_shape for s in self.sources]
        ys, xs = zip(*block_shapes)
        self._block_shape = (max(ys), max(xs))
        return self._block_shape

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @property
    def mask(self):
        return self._mask

    @property
    def what(self):
        return self._what

    @property
    def msgs(self):
        return self._msgs

    @msgs.setter
    def msgs(self, val):
        self._msgs = val
