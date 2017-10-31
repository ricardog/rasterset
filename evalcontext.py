import numpy as np
import numpy.ma as ma
import rasterio
import rasterio.features

def window_shape(win):
  return (win[0][1] - win[0][0], win[1][1] - win[1][0])

class EvalContext(object):
  def __init__(self, rasterset, what, crop=True):
    self._rasterset = rasterset
    self._what = what
    self._crop = crop
    self._msgs = True
    self._mask = None
    self._nodata = -9999
    self._needed = rasterset.find_needed(what)
    self._sources = [rasterset[x].source for x in
                     filter(lambda c: rasterset[c].is_raster, self._needed)]

    # Check all rasters have the same resolution.
    # TODO:: scale rasters appropriatelly
    self._res, self._crs = self.check_rasters(self.sources)
    # Compute reading window and affine transform for every raster
    self._bounds, self._affine, self._shape = self.find_bounds(self.sources)

    # Update bounds, affine, and shape if mask given
    if rasterset.shapes or rasterset.mask:
      self._bounds, self._affine, \
        self._shape, self._mask = self.apply_mask(rasterset.shapes,
                                                  rasterset.mask,
                                                  rasterset.maskval,
                                                  rasterset.all_touched)
      
    # Compute minimal block shape that covers all block shapes
    self._block_shape = self.block_shape(self.sources)

  @staticmethod
  def check_rasters(columns):
    readers = tuple(map(lambda c: c.reader, columns))

    first = readers[0]
    first_res = first.res
    first_crs = first.crs
    # Verify all rasters either have same CRS or don't have a CRS
    for reader in readers:
      if first_crs.to_string() == '' and reader.crs.to_string() != '':
        first_crs = reader.crs
      elif reader.crs.to_string() == '':
        pass
      elif first_crs != reader.crs:
        raise RuntimeError('%s: crs mismatch (%s != %s)' % (reader.name,
                                                            first_crs,
                                                            reader.crs))
      if not np.allclose(first_res, reader.res):
        raise RuntimeError('%s: resolution mismatch (%s != %s)' % (reader.name,
                                                                   first_res,
                                                                   reader.res))
    return first_res, first_crs

  @staticmethod
  def mask_bounds(shapes):
    if shapes is None:
      return None
    all_bounds = [rasterio.features.bounds(shape) for shape in shapes]
    minxs, minys, maxxs, maxys = zip(*all_bounds)
    bounds = (min(minxs), min(minys), max(maxxs), max(maxys))
    return bounds
    
  @staticmethod
  def compute_mask(shapes, all_touched, affine, shape):
    if shapes is None:
      return
    gshapes = [feature["geometry"] for feature in shapes]
    mask = rasterio.features.geometry_mask(gshapes,
                                           transform = affine,
                                           invert = False,
                                           out_shape = shape,
                                           all_touched = all_touched)
    return mask

  def apply_mask(self, shapes, mask_ds, maskval=1.0, all_touched=True):
    if shapes:
      mask_bounds = self.mask_bounds(shapes)
    else:
      mask_bounds = mask_ds.bounds
    if rasterio.coords.disjoint_bounds(self.bounds, mask_bounds):
      raise ValueError("rasters do not intersect mask")
    if self._crop:
      minxs, minys, maxxs, maxys = zip(self.bounds, mask_bounds)
      bounds = (max(minxs), max(minys), min(maxxs), min(maxys))
    else:
      bounds = self.bounds
    window = self.sources[0].reader.window(*bounds)
    affine = self.sources[0].reader.window_transform(window)
    shape = window_shape(window)
    if not mask_ds:
      mask = self.compute_mask(shapes, all_touched,
                               affine, shape)
    else:
      win = mask_ds.window(*bounds)
      data = mask_ds.read(1, masked=True, window=win)
      mask = ma.where(data == maskval, True, False)
    return bounds, affine, shape, mask
  
  @staticmethod
  def find_bounds(columns):
    sources = list(map(lambda c: c.reader, columns))
    first = sources.pop(0)
    bounds = list(tuple(first.bounds))

    for src in sources:
      src_bounds = src.bounds
      if rasterio.coords.disjoint_bounds(bounds, src_bounds):
        raise ValueError("rasters do not intersect")
      bounds[0] = max(bounds[0], src_bounds[0])
      bounds[1] = max(bounds[1], src_bounds[1])
      bounds[2] = min(bounds[2], src_bounds[2])
      bounds[3] = min(bounds[3], src_bounds[3])
    # The number of rows, columns must be the same for all rasters.
    # Compute shape based using first raster.
    bounds = tuple(bounds)
    window = first.window(*bounds)
    affine = first.window_transform(window)
    return bounds, affine, window_shape(window)
  
  @staticmethod
  def block_shape(sources):
    block_shapes = [s.block_shape for s in sources]
    ys, xs = zip(*block_shapes)
    block_shape = (max(ys), max(xs))
    blocks = set(block_shapes)
    return block_shape
  
  def block_windows(self):
    y_inc, x_inc = self._block_shape
    for j in range(0, self.height, y_inc):
      j2 = min(j + y_inc, self.height)
      for i in range(0, self.width, x_inc):
        i2 = min(i + x_inc, self.width)
        yield ((j, j2), (i, i2))

  def need(self, what):
    return what in self._needed

  def meta(self, args={}):
    meta = self.sources[0].reader.meta.copy()
    meta.update({'driver': 'GTiff', 'compress': 'lzw', 'predictor': 2,
                 'nodata': self._nodata})
    meta.update(args)
    meta.update({'count': 1, 'crs': self._crs,
                 'transform': self.affine,
                 'height': self.height, 'width': self.width,
                 'dtype': np.float32})
    return meta
  
  @property
  def sources(self):
    return self._sources

  @property
  def bounds(self):
    return self._bounds

  @property
  def affine(self):
    return self._affine

  @property
  def shape(self):
    return self._shape

  @property
  def height(self):
    return self._shape[0]

  @property
  def width(self):
    return self._shape[1]

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
