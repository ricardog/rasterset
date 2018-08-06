
from .raster import Raster
from ..simpleexpr import SimpleExpr

class RasterCol(object):
  def __init__(self, name, obj, mask, bbox):
    self._name = name
    self._obj = obj
    self._mask = mask
    self._bbox = bbox
    self._series = None
    if getattr(obj, 'eval', None) is None:
      raise RuntimeError("column %s does not have 'eval' callback" % name)
    self._eval = obj.eval
    if getattr(obj, 'syms', None) is None:
      raise RuntimeError("column %s does not have 'syms' callback" % name)
    self._inputs = set(obj.syms)

  @property
  def name(self):
    return self._name

  @property
  def mask(self):
    return self._mask

  @property
  def bbox(self):
    return self._bbox

  @property
  def inputs(self):
    return self._inputs

  @property
  def is_raster(self):
    return isinstance(self._obj, Raster)

  @property
  def source(self):
    return self._obj

  @property
  def data(self):
    return self._series

  @data.setter
  def data(self, series):
    self._series = series

  def eval(self, df, window=None):
    if isinstance(self._obj, (Raster, SimpleExpr)):
      return self._eval(df, window)
    else:
      return self._eval(df)
