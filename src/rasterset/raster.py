import numpy.ma as ma
from pathlib import Path
import rasterio.windows as rwindows
import rioxarray as rxr
from urllib.parse import urlparse

from . import window

class Raster(object):
    def __init__(self, fname, band=1):
        self._fname = fname
        self._rxr = None
        self._band = band
        self._mask = None
        self._bbox = None
        self._window = None
        self._str = None

    @property
    def syms(self):
        return []

    @property
    def inputs(self):
        return set()

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    @property
    def bbox(self):
        return self._bbox

    @property
    def res(self):
        return self.reader.rio.resolution()

    @property
    def nodata(self):
        return self._rxr.attrs['_FillValue']

    @property
    def bounds(self):
        return self.reader.rio.bounds()

    @property
    def transform(self):
        return self.reader.rio.transform()

    @property
    def crs(self):
        try:
            return self.reader.rio.crs
        except rxr.exceptions.MissingCRS:
            return None

    @property
    def dtype(self):
        return self._rxr.dtype

    @property
    def shape(self):
        return self._rxr[self._band - 1, ...].shape

    @property
    def array(self):
        return self._rxr

    @property
    def reader(self):
        if self._rxr is None:
            self._rxr = self.open()
        return self._rxr

    def open(self):
        return rxr.open_rasterio(self._fname, parse_coordinates=False,
                                 chunks='auto')

    def clip(self, bounds):
        self._window = window.round(rwindows.from_bounds(*bounds,
                                                         self.transform))
        #import pdb; pdb.set_trace()
        self._rxr = self.reader.rio.isel_window(self._window)
        return

    def eval(self, df, win=None):
        if win is None:
            win = self._window
        data = self.reader.rio.isel_window(win)[self._band - 1]
        mdata = ma.masked_equal(data.to_masked_array(copy=False),
                                self.nodata)
        mdata.mask = mdata.mask | self.mask[win.toslices()]
        return mdata

    def __repr__(self):
        fname = str(self._fname)
        parts = urlparse(fname)
        if parts.scheme is None:
            path = Path(parts.path)
            if path.is_absolute():
                return path.as_uri()
            return fname
        return fname

    def __str__(self):
        if self._str is None:
            fname = str(self._fname)
            parts = urlparse(fname)
            path = Path(parts.path)
            scheme = parts.scheme or "file"
            if len(path.parts) > 2:
                short = Path(*("...",) + path.parts[-2:])
            else:
                short = path
            if parts.netloc == "":
                self._str = f"{scheme}://{short}"
            else:
                self._str = f"{scheme}://{parts.netloc}/{short}"
        return self._str
