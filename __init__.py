
import click
import concurrent.futures
from functools import reduce
import math
import multiprocessing
import numpy as np
import numpy.ma as ma
import rasterio
from tqdm import tqdm

from .evalcontext import EvalContext
from .raster import Raster
from .rastercol import RasterCol

class RasterSet(object):
  def __init__(self, data=None, shapes=None, bbox=None, mask=None,
               maskval=1.0, crop=True, all_touched=False):
    self._mask = mask
    self._maskval = maskval
    self._shapes = shapes
    if mask and shapes:
      raise RuntimeError('specify only mask or shapefile mask')
    self._bbox = bbox
    self._crop = crop
    self._all_touched = all_touched
    self._levels = []
    if data is None:
      self._data = {}
    elif isinstance(data, dict):
      self._data = {}
      for k, v in data.items():
        self._data[k] = RasterCol(k, v, None, bbox)
    else:
      raise RuntimeError('unknown data source')

  def __getitem__(self, key):
    if key in self._data:
      return self._data[key]
    raise KeyError(key)

  def __setitem__(self, key, value):
    self._data[key] = RasterCol(key, value, self.mask, self.bbox)
    self._levels = []

  def __contains__(self, key):
    return key in self._data

  def keys(self):
    return self._data.keys()
  
  @property
  def mask(self):
    return self._mask

  @property
  def maskval(self):
    return self._maskval

  @property
  def shapes(self):
    return self._shapes

  @shapes.setter
  def shapes(self, shapes):
    self._shapes = shapes

  @property
  def bbox(self):
    return self._bbox

  @property
  def crop(self):
    return self._crop

  @crop.setter
  def crop(self, crop):
    self._crop = crop

  @property
  def all_touched(self):
    return self._all_touched

  @property
  def levels(self):
    if not self._levels:
      self.compute_order()
    return self._levels

  def compute_order(self):
    '''Compute a partial ordering of the rasters.  Checks for cycles in the
    graph as it does the work.'''
    if self._levels:
      return
    order = {}
    visiting = {}
    def visit(col):
      if col.name in order:
        return
      me = 0
      if col.name in visiting and col.name not in order:
        raise RuntimeError('circular dependency')
      visiting[col.name] = True
      for name in col.inputs:
        if name not in order:
          visit(self._data[name])
        me = max(me, order[name] + 1)
      del visiting[col.name]
      order[col.name] = me
    for col in self._data.values():
      visit(col)

    ordered = sorted(order.items(), key=lambda kv: (kv[1], kv[0]))
    nlevels = ordered[-1][1]
    self._levels = [[] for x in range(nlevels + 1)]
    for k, v in ordered:
      self._levels[v].append(k)
    self._levels[0].sort(key=lambda a: -1 if isinstance(self._data[a].source,
                                                        Raster) else 1)

  def to_dot(self):
    import pdb; pdb.set_trace()
    print("digraph: LogAbund {")
    print("node [fontname: Palatino, fontsize: 24];")
    for idx, level in enumerate(self._levels):
      for name in level:
        print('"%s" [];' % name)
        for dep in self[name].inputs:
          print('"%s" -> "%s"' % (dep, name))
    print("}")
    
  
  def set_props(self, ctx):
    for col in ctx.sources:
      if ctx.mask is not None:
        col.mask = ctx.mask
      col.window = col.reader.window(*ctx.bounds)

  def find_needed(self, what):
    def transitive(me):
      ret = list(me.inputs)
      for name in me.inputs:
        if name in self:
          ret += transitive(self[name])
        else:
          raise KeyError(name)
      return ret

    if what not in self._data:
      raise KeyError(what)
    return set(transitive(self[what]) + [what])

  def dropna(self, df):
    namask = reduce(np.logical_or, map(ma.getmask, df.values()),
                    np.zeros(tuple(df.values())[0].shape))
    for key in df.keys():
      df[key].mask = namask
      df[key] = df[key].compressed()
    return namask

  def reflate(self, namask, data):
    arr = ma.empty_like(namask, dtype=np.float32)
    arr.mask = namask
    arr[~namask] = data
    return arr
    
  def _eval(self, ctx, window=None):
    # Compute partial order in which to evaluate rasters
    self.compute_order()

    df = {}
    for idx, level in enumerate(self._levels):
      if ctx.msgs:
        click.echo("Level %d" % idx)
      for name in level:
        if ctx.need(name):
          if ctx.msgs:
            click.echo("  eval %s" % name)
          df[name] = self[name].eval(df, window)
      if idx == 0:
        namask = self.dropna(df)
    data = ma.empty_like(namask, dtype=np.float32)
    data.mask = namask
    data[~namask] = df[ctx.what]
    if False:
      import pandas as pd
      dframe = pd.DataFrame(df)
      import pd_utils
      pd_utils.save_pandas('evaled.pyd', dframe)
    if False:
      import pandas as pd
      dframe = pd.DataFrame(df)
      import projections.pd_utils as pd_utils
      df2 = {}
      df3 = {}
      for k in df.keys():
        tmp = ma.empty_like(namask, dtype=np.float32)
        tmp.mask = namask
        tmp[~namask] = df[k]
        df2[k] = tmp#.reshape(-1)
        df3[k] = df2[k][75:135, 880]
      dframe = pd.DataFrame(df3)
      #pd_utils.save_pandas('evaled.pyd', dframe)
      dframe.to_csv('evaled.csv')
      import pdb; pdb.set_trace()
      #import pandas as pd
      #dframe = pd.DataFrame(df2)
      #import pd_utils
      #pd_utils.save_pandas('1950.pyd', dframe)
    return data

  def eval(self, what, quiet=False, args={}):
    ctx = EvalContext(self, what, crop=self.crop)
    if quiet:
      ctx.msgs = False
    self.set_props(ctx)
    data = self._eval(ctx)
    meta = ctx.meta(args)
    data.set_fill_value(meta['nodata'])
    return data, meta
  
  def swrite(self, what, path, crop=True, args={}):
    ctx = EvalContext(self, what)
    self.set_props(ctx)
    meta = ctx.meta(args)
    iters = math.ceil(1.0 * ctx.height / ctx._block_shape[0])
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_CACHEMAX=256):
      with rasterio.open(path, 'w', **meta) as dst:
        with click.progressbar(ctx.block_windows(), iters) as bar:
          for win in bar:
            height, width = (win[0][1] - win[0][0], win[1][1] - win[1][0])
            out = self._eval(ctx, win)
            dst.write(out.filled(meta['nodata']), window = win, indexes = 1)
            ctx.msgs = False


  def write(self, what, path, crop=True, args={}):
    ctx = EvalContext(self, what)
    self.set_props(ctx)
    meta = ctx.meta(args)
    # by default ThreadPoolExecutor uses num_cpus() * 5 but that's too
    # high for this problem because threads start competing for the GIL.
    # Increasing the clock size helps, at the cost of higher memory
    # utilization.
    num_workers = multiprocessing.cpu_count()
    ctx.msgs = False

    def jobs():
      for win in ctx.block_windows():
        yield win

    def compute(win):
      return self._eval(ctx, win)

    bar = tqdm(leave=True, total=ctx.height, desc="projecting")
    
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_CACHEMAX=256):
      with rasterio.open(path, 'w', **meta) as dst:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers) as executor:
          future_to_window = {
            executor.submit(compute, win): win for win in jobs()
          }
          for future in concurrent.futures.as_completed(future_to_window):
            win = future_to_window[future]
            rows = win[0][1] - win[0][0]
            bar.update(rows)
            out = future.result()
            dst.write(out.filled(meta['nodata']), window = win, indexes = 1)
    bar.close()
