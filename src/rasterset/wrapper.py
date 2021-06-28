
from affine import Affine
import numpy as np
from rasterio.crs import CRS
from rasterio.profiles import DefaultGTiffProfile
import rasterio.windows as rwindows
from xarray import Dataset
from dask.array.core import broadcast_chunks
from xarray.backends.common import BackendArray
from xarray.core import indexing
from xarray.core.utils import is_scalar

from .mask import mask_maker
from .raster import Raster
from .simpleexpr import SimpleExpr
from . import bounds


WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90)
EPSG_4326 = CRS.from_epsg(4326)


def is_raster(x):
    return isinstance(x, Raster)


def is_constant(x):
    return isinstance(x, SimpleExpr) and (x.is_constant is not None)


def is_source(x):
    return is_raster(x) or is_constant(x)


def _check_alignment(src):
    x_off = src.transform[2] % src.transform[0]
    y_off = src.transform[5] % abs(src.transform[4])
    xx = x_off < 1e-10 or (abs(x_off - src.transform[0]) < 1e-10)
    yy = y_off < 1e-10 or (abs(y_off - abs(src.transform[4])) < 1e-10)
    return xx and yy


def _compute_order(needed, items):
    """Compute a partial ordering for evaluation of functions.  Checks for
    cycles in the graph as it does the work.

    """
    order = {}
    visiting = {}
    sources = dict(items)

    def visit(name, col):
        if name in order:
            return
        me = 1
        if name in visiting and name not in order:
            raise RuntimeError("circular dependency")
        visiting[name] = True
        for other in col.inputs:
            if other not in order:
                visit(other, sources[other])
            me = max(me, order[other] + 1)
        del visiting[name]
        order[name] = me

    for name, col in items:
        if name in needed:
            visit(name, col)

    ordered = sorted(order.items(), key=lambda kv: (kv[1], kv[0]))
    nlevels = ordered[-1][1]
    levels = [[] for x in range(nlevels + 1)]
    for k, v in ordered:
        if not is_raster(sources[k]):
            levels[v].append(k)
    levels[0] = tuple(filter(lambda k: is_raster(sources[k]), needed))
    return levels


class RastersetArrayWrapper(BackendArray):
    """A wrapper around rasterio dataset objects"""
    def __init__(self, name, rasterset, crop=True, bbox=None, dtype='float32'):
        self._name = name
        self._crop = crop
        self._dtype = np.dtype(dtype)
        self._crs = None
        self._mask = None
        self._bounds = None
        self._transform = None
        self._fill_value = self._dtype.type(-9999.0)
        self._window = None
        self._shape = None
        self._block_shape = None
        self._needed = sorted(rasterset.find_needed(name), key=lambda x: x.lower())
        self._sources = tuple(filter(lambda c: is_raster(c),
                                     [rasterset[x] for x in self._needed]))
        assert self._sources, "No rasters of know size in input set"

        # Check all rasters have same CRS and resolution
        if len(set(map(lambda src: src.crs, self._sources))) != 1:
            raise RuntimeError("All rasters must have the same CRS")
        self._crs = self._sources[0].crs
        if len(set(map(lambda src: src.res, self._sources))) != 1:
            raise RuntimeError("All rasters must have the same resolution")
        res = self._sources[0].res

        # Check all rasters are alligned.
        for src in self._sources:
            if not _check_alignment(src):
                raise RuntimeError(f"Raster {src} is not aligned")
        # Compute bounds as intersection of all raster bounds.
        self._bounds = bounds.intersection(bbox or WORLD_BOUNDS,
                                           *(s.bounds for s in self._sources))

        self._mask = mask_maker(rasterset.shapes, rasterset.all_touched,
                                rasterset.mask, rasterset.maskval)
        # Crop output to bounds of mask.
        if crop:
            self._bounds = bounds.intersection(self.bounds, self._mask.bounds)

        # Set the affine transform and calculate mask.
        self._transform = (Affine.translation(self.bounds.left,
                                              self.bounds.top) *
                           Affine.scale(res[0], res[1]) *
                           Affine.identity())
        self._mask.transform = self.transform
        self._mask.eval(self.bounds)
        self._dataset = Dataset()
        self._columns = {}
        for name, src in rasterset.items():
            if name in self._needed:
                if is_raster(src):
                    src = src.clip(self.bounds)
                self._columns[name] = src
        self._sources = [src for src in self._columns.values() if is_raster(src)]
        chunkss = [src.chunks for src in self._sources]
        shapes = [src.asarray().shape for src in self._sources]
        rasters = [(name, src) for name, src in self._columns.items()
                   if is_raster(src)]
        assert len(set([ras.asarray().dims for ras in self._sources])) == 1
        self._dataset = Dataset(dict(rasters))
        self._chunks = broadcast_chunks(chunkss)[0]
        self._shape = np.broadcast_shapes(*shapes)
        self._dims = self._sources[0].asarray().dims
        self._coords = self._sources[0].asarray().coords
        self._levels = _compute_order(self._needed, self._columns.items())

#        self._dataset = self._dataset.update(
#            dict([(name, DataArray(np.broadcast_to(src.asarray(), self.shape,
#                                                   subok=True),
#                                   dims=self.dims, coords=self.coords))
#                  for name, src in rasterset.items()
#                  if is_constant(src)]))
        return

    @property
    def dtype(self):
        """
        Data type of the array
        """
        return self._dtype

    @property
    def fill_value(self):
        """
        Fill value of the array
        """
        return self._fill_value

    @property
    def shape(self):
        """
        Shape of the array
        """
        return self._shape

    @property
    def bounds(self):
        """Return the bounds of the raster."""
        return self._bounds

    @property
    def transform(self):
        """Return the transform of the raster."""
        return self._transform

    @property
    def crs(self):
        """Return the CRS of the raster."""
        return self._crs

    @property
    def chunks(self):
        """Return the chunks of the raster."""
        return self._chunks

    @property
    def res(self):
        """Return the resolution of the raster."""
        return (self.transform.a, self.transform.e)

    @property
    def height(self):
        return self.shape[1]

    @property
    def width(self):
        return self.shape[2]

    @property
    def dims(self):
        return self._dims

    @property
    def coords(self):
        return self._coords

    def dataset(self):
        return self._dataset

    def meta(self, args={}):
        meta = DefaultGTiffProfile(count=1,
                                   dtype=np.float32,
                                   predictor=3,
                                   crs=self.crs,
                                   nodata=self.fill_value,
                                   transform=self.transform,
                                   width=self.width,
                                   height=self.height,
                                   sparse_ok="YES",
                                   )
        meta.update(args)
        return meta

    def _get_indexer(self, key):
        """Get indexer for rasterio array.

        Parameter
        ---------
        key: tuple of int

        Returns
        -------
        band_key: an indexer for the 1st dimension
        window: two tuples. Each consists of (start, stop).
        squeeze_axis: axes to be squeezed
        np_ind: indexer for loaded numpy array

        See also
        --------
        indexing.decompose_indexer
        """
        if len(key) != 3:
            raise ValueError("rasterio datasets should always be 3D")

        # bands cannot be windowed but they can be listed
        band_key = key[0]
        np_inds = []
        # bands (axis=0) cannot be windowed but they can be listed
        if isinstance(band_key, slice):
            start, stop, step = band_key.indices(self.shape[0])
            band_key = np.arange(start, stop, step)
        # be sure we give out a list
        band_key = (np.asarray(band_key) + 1).tolist()
        if isinstance(band_key, list):  # if band_key is not a scalar
            np_inds.append(slice(None))

        # but other dims can only be windowed
        window = []
        squeeze_axis = []
        for iii, (ikey, size) in enumerate(zip(key[1:], self.shape[1:])):
            if isinstance(ikey, slice):
                # step is always positive. see indexing.decompose_indexer
                start, stop, step = ikey.indices(size)
                np_inds.append(slice(None, None, step))
            elif is_scalar(ikey):
                # windowed operations will always return an array
                # we will have to squeeze it later
                squeeze_axis.append(-(2 - iii))
                start = ikey
                stop = ikey + 1
            else:
                start, stop = np.min(ikey), np.max(ikey) + 1
                np_inds.append(ikey - start)
            window.append((start, stop))

        if isinstance(key[1], np.ndarray) and isinstance(key[2], np.ndarray):
            # do outer-style indexing
            np_inds[-2:] = np.ix_(*np_inds[-2:])

        return band_key, tuple(window), tuple(squeeze_axis), tuple(np_inds)

    def eval(self, key, window):
        df = {}
        for idx, level in enumerate(self._levels):
            for name in level:
                if idx == 0:
                    win = rwindows.Window.from_slices(
                        rows=window[0], cols=window[1],
                        height=self.shape[1], width=self.shape[2])
                    mdata = self._columns[name].eval(win=win)
                    mdata.mask = mdata.mask | self._mask.mask[win.toslices()]
                    df[name] = mdata
                else:
                    df[name] = self._columns[name].eval(df)
        return df[self._name]

    def _getitem(self, key):
        band_key, window, squeeze_axis, np_inds = self._get_indexer(key)
        out = self.eval(key, window)

        if squeeze_axis:
            out = np.squeeze(out, axis=squeeze_axis)
        return out[np_inds].filled(self.fill_value)

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )
