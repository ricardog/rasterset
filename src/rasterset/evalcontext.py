import numpy as np
import numpy.ma as ma
import rasterio
import rasterio.features

from . import window


WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90)
class EvalContext(object):
    def __init__(self, rasterset, what, crop=True, bbox=None):
        self._rasterset = rasterset
        self._what = what
        self._crop = crop
        self._msgs = True
        self._mask = None
        self._bounds = None
        self._transform = None
        self._nodata = -9999
        self._needed = sorted(rasterset.find_needed(what), key=lambda x: x.lower())
        self._sources = tuple(filter(lambda c: c.is_raster,
                                     [rasterset[x] for x in self._needed]))

        # Check all rasters have the same resolution.
        # TODO:: scale rasters appropriatelly
        self._res, self._crs = self.check_rasters(self.sources)
        # Compute reading window and affine transform for every raster
        self.bounds = window.intersection(
            bbox or WORLD_BOUNDS,
            *(s.reader.bounds for s in self.sources)
        )

        # Update bounds, transform, and shape if mask given
        if rasterset.shapes or rasterset.mask:
            self.bounds, self.mask = self.apply_mask(
                rasterset.shapes,
                rasterset.mask,
                rasterset.maskval,
                rasterset.all_touched,
            )

        # Set the window & mask for all sources
        for src in self.sources:
            src.window = window.round(src.reader.window(*self.bounds))
            if self.mask is not None:
                src.mask = self.mask

        # The number of rows and columns must be the same for all rasters.
        # Trigger and assert if there is more than one window shape in
        # the raster set.
        shapes = [window.shape(src.window) for src in self.sources]
        assert len(set(shapes)) == 1, "More than one window size"

        # Set the affine transform and shape for the output
        self._transform = self.sources[0].reader.window_transform(
            self.sources[0].window
        )
        self._shape = window.shape(self.sources[0].window)
        if self.mask is not None:
            assert self.shape == self.mask.shape

        # Compute minimal block shape that covers all block shapes
        self._block_shape = self.block_shape(self.sources)

    @staticmethod
    def check_rasters(columns):
        def check_alignment(src):
            x_off = src.transform[2] % src.transform[0]
            y_off = src.transform[5] % abs(src.transform[4])
            xx = x_off < 1e-10 or (abs(x_off - src.transform[0]) < 1e-10)
            yy = y_off < 1e-10 or (abs(y_off - abs(src.transform[4])) < 1e-10)
            return xx and yy

        readers = tuple(map(lambda c: c.reader, columns))

        assert readers, "No rasters of know size in input set"

        first = readers[0]
        first_res = first.res
        first_crs = first.crs
        # Verify all rasters either have same CRS or don't have a CRS
        for reader in readers:
            if (first_crs is None or first_crs.to_string() == "") and (
                reader.crs is not None and reader.crs.to_string() != ""
            ):
                first_crs = reader.crs
            elif reader.crs is None or reader.crs.to_string() == "":
                pass
            elif first_crs != reader.crs:
                raise RuntimeError(
                    "%s: crs mismatch (%s != %s)" % (reader.name, first_crs, reader.crs)
                )
            if not np.allclose(first_res, reader.res):
                raise RuntimeError(
                    "%s: resolution mismatch (%s != %s)"
                    % (reader.name, first_res, reader.res)
                )
        for col in columns:
            if not check_alignment(col.reader):
                print("WARNING: raster %s has unaligned pixels" % col.name)

        return first_res, first_crs

    @staticmethod
    def mask_bounds(shapes):
        if shapes is None:
            return WORLD_BOUNDS
        bounds = getattr(shapes, 'bounds', None)
        if bounds:
            return bounds
        return window.union(shapes)

    @staticmethod
    def rasterize_mask(shapes, all_touched, transform, shape):
        if shapes is None:
            return
        gshapes = [feature["geometry"] for feature in shapes]
        mask = rasterio.features.geometry_mask(
            gshapes,
            transform=transform,
            invert=False,
            out_shape=shape,
            all_touched=all_touched,
        )
        return mask

    def apply_mask(self, shapes, mask_ds, maskval=1.0, all_touched=True):
        if shapes:
            mbounds = self.mask_bounds(shapes)
        else:
            mbounds = mask_ds.bounds
        if rasterio.coords.disjoint_bounds(self.bounds, mbounds):
            raise ValueError("rasters do not intersect mask")
        if self._crop:
            crop_bounds = window.intersection(self.bounds, mbounds)
        else:
            crop_bounds = self.bounds

        if not mask_ds:
            win = window.round(self.sources[0].reader.window(*crop_bounds))
            mask = self.rasterize_mask(
                shapes,
                all_touched,
                self.sources[0].reader.window_transform(win),
                window.shape(win)
            )
        else:
            win = window.round(mask_ds.window(*crop_bounds))
            data = mask_ds.read(1, masked=True, window=win)
            mask = ma.where(data == maskval, True, False).filled(True)
        return crop_bounds, mask

    @staticmethod
    def block_shape(sources):
        block_shapes = [s.block_shape for s in sources]
        ys, xs = zip(*block_shapes)
        block_shape = (max(ys), max(xs))
        return block_shape

    def block_windows(self):
        y_inc, x_inc = self._block_shape
        for j in range(0, self.height, y_inc):
            j2 = min(j + y_inc, self.height)
            for i in range(0, self.width, x_inc):
                i2 = min(i + x_inc, self.width)
                yield Window(col_off=i, row_off=j,
                             width=i2 - i, height=j2 - j)

    def need(self, what):
        return what in self._needed

    def meta(self, args={}):
        meta = self.sources[0].reader.meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "compress": "lzw",
                "predictor": 3,
                "nodata": self._nodata,
                "sparse_ok": "YES",
            }
        )
        meta.update(args)
        meta.update(
            {
                "count": 1,
                "crs": self._crs,
                "transform": self.transform,
                "height": self.height,
                "width": self.width,
                "dtype": np.float32,
            }
        )
        return meta

    @property
    def sources(self):
        return self._sources

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        self._bounds = bounds

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def height(self):
        return self._shape[0]

    @property
    def width(self):
        return self._shape[1]

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    @property
    def what(self):
        return self._what

    @property
    def msgs(self):
        return self._msgs

    @msgs.setter
    def msgs(self, val):
        self._msgs = val
