import numpy as np
import numpy.ma as ma
import rasterio
import rasterio.errors
import threading


def window_inset(win1, win2):
  if win2:
    return ((win1[0][0] + win2[0][0], min(win1[0][0] + win2[0][1], win1[0][1])),
            (win1[1][0] + win2[1][0], min(win1[1][0] + win2[1][1], win1[1][1])))
  return win1

class Raster(object):
  def __init__(self, name, fname, band=1):
    self._name = name
    self._fname = fname
    self._threadlocal = threading.local()
    self._band = band
    self._window = None
    self._affine = None
    self._mask = None

  @property
  def name(self):
    return self._name

  @property
  def syms(self):
    return []

  @property
  def reader(self):
    if getattr(self._threadlocal, 'reader', None) is None:
      try:
        self._threadlocal.reader = rasterio.open(self._fname)
      except (SystemError, rasterio.errors.RasterioIOError) as e:
        print("Error: opening raster '%s' for %s" % (self._fname, self.name))
        raise SystemError("Error: opening raster '%s' for %s" %
                          (self._fname, self.name))
    return self._threadlocal.reader

  @property
  def window(self):
    return self._window

  @window.setter
  def window(self, window):
    self._window = window

  @property
  def affine(self):
    return self._affine

  @affine.setter
  def affine(self, affine):
    self._affine = affine

  @property
  def mask(self):
    return self._mask

  @mask.setter
  def mask(self, mask):
    self._mask = mask

  @property
  def block_shape(self):
    return self.reader.block_shapes[self._band - 1]

  @property
  def dtype(self):
    return self.reader.dtypes[self._band - 1]
  
  def eval(self, df, window=None):
    assert self.window
    win = window_inset(self.window, window)
    try:
      data = self.reader.read(self._band, window=win, masked=True)
    except IndexError as e:
      print("Error: reading band %d from %s" % (self._band, self._fname))
      raise IndexError("Error: reading band %d from %s" %
                       (self._band, self._fname))
    ## HPD raster sometimes has NODATA values that leak.
    data = ma.where(data < -1e20, np.nan, data)
    if self.mask is not None:
      if window:
        data.mask = data.mask | self.mask[window[0][0]:window[0][1],
                                          window[1][0]:window[1][1]]
      else:
        data.mask = data.mask | self.mask
    return data
