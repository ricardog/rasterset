from pathlib import Path
import threading
import rasterio
import rasterio.errors
from rasterio.windows import intersection
from urllib.parse import urlparse


def window_inset(win1, win2):
    if win2:
        return intersection(win1, win2)
    return win1


class Raster(object):
    def __init__(self, fname, band=1):
        self._fname = fname
        self._threadlocal = threading.local()
        self._band = band
        self._window = None
        self._mask = None
        self._bbox = None
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
    def reader(self):
        if getattr(self._threadlocal, "reader", None) is None:
            try:
                self._threadlocal.reader = rasterio.open(self._fname)
            except (SystemError, rasterio.errors.RasterioIOError):
                print("Error: opening raster '%s'" % (self._fname))
                raise SystemError("Error: opening raster '%s'" % (self._fname))
        return self._threadlocal.reader

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        self._window = window

    @property
    def block_shape(self):
        return self.reader.block_shapes[self._band - 1]

    @property
    def dtype(self):
        return self.reader.dtypes[self._band - 1]

    def eval(self, df, window=None):
        assert self.window
        if window:
            win = intersection(self.window, window)
        else:
            win = self.window
        try:
            data = self.reader.read(self._band, window=win, masked=True)
        except IndexError:
            print("Error: reading band %d from %s" % (self._band, self._fname))
            raise IndexError(
                "Error: reading band %d from %s" % (self._band, self._fname)
            )
        if self.mask is not None:
            if window:
                data.mask = (
                    data.mask | self.mask[window.toslices()]
                )
            else:
                data.mask = data.mask | self.mask
        return data

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
