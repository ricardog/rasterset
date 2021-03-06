
from collections import OrderedDict
import concurrent.futures
from functools import reduce
import math
import multiprocessing
from pprint import pprint

import asciitree
import click
import dask.array as da
from dask.base import tokenize
import numpy as np
import numpy.ma as ma
import rasterio
import rioxarray as rxr
from tqdm import tqdm
import xarray as xa
from xarray.core import indexing
from xarray.core.dataarray import DataArray

from .evalcontext import EvalContext
from .raster import Raster
from .simpleexpr import SimpleExpr
from .wrapper import RastersetArrayWrapper


def is_raster(x):
    return isinstance(x, Raster)


def is_constant(x):
    return isinstance(x, SimpleExpr) and (x.is_constant is not None)


class RasterSet(object):
    def __init__(
        self,
        data=None,
        shapes=None,
        bbox=None,
        mask=None,
        maskval=1.0,
        crop=True,
        all_touched=False,
    ):
        self._mask = mask
        self._maskval = maskval
        self._shapes = shapes
        if mask and shapes:
            raise RuntimeError("specify only mask or shapefile mask")
        self._bbox = bbox
        self._crop = crop
        self._all_touched = all_touched
        self._dtype = np.dtype('float32')                   # FIXME
        if data is None:
            self._data = {}
        elif isinstance(data, dict):
            self._data = {}
            for k, v in data.items():
                self.__setitem__(k, v)
        else:
            raise RuntimeError("unknown data source")

    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        # FIXME: this generates an error when computing order.
        # if isinstance(value, str) and value in self._data:
        #  self._data[key] = self._data[value]
        #  return
        if isinstance(value, (int, float)):
            value = SimpleExpr(value)
        elif isinstance(value, str):
            value = SimpleExpr(value)
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]
        return

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def update(self, other, fname=None):
        if fname is None:
            fname = '/dummy/file/name.nc'
        if isinstance(other, xa.Dataset) and hasattr(other, 'rio'):
            for key, data in other.items():
                self[key] = Raster.from_dataarray(data)
        elif isinstance(other, dict):
            self._data.update(other)
        else:
            raise RuntimeError("'other' in update must be dict or "
                               "rioxarray Dataset")
        return

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
    def dtype(self):
        return self._dtype

    def compute_order(self, what):
        """Compute a partial ordering of the rasters.  Checks for cycles
        in the graph as it does the work.

        """
        order = {}
        visiting = {}
        needed = self.find_needed(what)

        def visit(name, col):
            if name in order:
                return
            me = 1
            if name in visiting and name not in order:
                raise RuntimeError("circular dependency")
            visiting[name] = True
            for other in col.inputs:
                if other not in order:
                    visit(other, self[other])
                me = max(me, order[other] + 1)
            del visiting[name]
            order[name] = me

        for name, col in self.items():
            if name in needed:
                visit(name, col)

        ordered = sorted(order.items(), key=lambda kv: (kv[1], kv[0]))
        nlevels = ordered[-1][1]
        levels = [[] for x in range(nlevels + 1)]
        for k, v in ordered:
            if not is_raster(self[k]):
                levels[v].append(k)
        levels[0] = tuple(filter(lambda k: is_raster(self[k]), needed))
        return levels

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
        shapes = []
        for col in ctx.sources:
            col.clip(ctx.bounds, inplace=True)
            shapes.append(col.shape)
        assert len(set(shapes)) == 1, "More than one window size"
        return

    def find_needed(self, what):
        def transitive(me):
            ret = list(me.inputs)
            for name in me.inputs:
                if name in self:
                    ret += transitive(self[name])
                else:
                    raise KeyError(name)
            return ret

        if what not in self:
            raise KeyError(what)
        return set(transitive(self[what]) + [what])

    def dropna(self, df):
        out_df = {}
        namask = reduce(
            np.logical_or,
            map(ma.getmask, df.values()),
            np.zeros(tuple(df.values())[0].shape),
        )
        for key in df.keys():
            df[key].mask = namask
            out_df[key] = df[key].compressed()
        return out_df, namask

    def reflate(self, namask, data):
        arr = ma.empty_like(namask, dtype=np.float32)
        arr.mask = namask
        arr[~namask] = data
        return arr

    def _eval(self, ctx, window=None):
        # Compute partial order in which to evaluate rasters
        levels = self.compute_order(ctx.what)

        df = {}
        for idx, level in enumerate(levels):
            if ctx.msgs:
                click.echo("Level %d" % idx)
            for name in level:
                if ctx.need(name):
                    if ctx.msgs:
                        click.echo("  eval %s" % name)
                    if idx == 0:
                        # First level sources must be able to subset the
                        # input via the window parameter.
                        mdata = self[name].eval(df, window)
                        if window:
                            mdata.mask = mdata.mask | ctx.mask.mask[window.toslices()]
                        else:
                            mdata.mask = mdata.mask | ctx.mask.mask
                        df[name] = mdata
                    else:
                        df[name] = self[name].eval(df)
            if idx == 0:
                df, namask = self.dropna(df)
        data = ma.empty_like(namask, dtype=np.float32)
        data.mask = namask
        data[~namask] = df[ctx.what]
        return data

    def build(self, what):
        wrapper = RastersetArrayWrapper(what, self, crop=self.crop,
                                        dtype=self.dtype)
        ds = wrapper.dataset()
        data = indexing.LazilyIndexedArray(wrapper)
        # FIXME: should the array be writeable?
        # data = indexing.CopyOnWriteArray(data)
        result = DataArray(data=data, dims=wrapper.dims,
                           coords=wrapper.coords, attrs=ds.attrs, name=what)
        result.attrs['_FillValue'] = wrapper.fill_value
        token = tokenize(what, wrapper.chunks)
        name_prefix = f"rasterset-{what}-{token}"
        return result.chunk(wrapper.chunks, name_prefix=name_prefix,
                            token=token), wrapper.meta()

    def eval(self, what, quiet=False, args={}):
        ctx = EvalContext(self, what, crop=self.crop, bbox=self.bbox)
        if quiet:
            ctx.msgs = False
        self.set_props(ctx)
        data = self._eval(ctx)
        meta = ctx.meta(args)
        data.set_fill_value(meta["nodata"])
        return data, meta

    def swrite(self, what, path, crop=True, args={}):
        ctx = EvalContext(self, what)
        self.set_props(ctx)
        meta = ctx.meta(args)
        iters = math.ceil(1.0 * ctx.height / ctx._block_shape[0])
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_CACHEMAX=256):
            with rasterio.open(path, "w", **meta) as dst:
                with click.progressbar(ctx.block_windows(), iters) as bar:
                    for win in bar:
                        out = self._eval(ctx, win)
                        dst.write(out.filled(meta["nodata"]), window=win, indexes=1)
                        ctx.msgs = False

    def write(self, what, path, crop=True, args={}):
        ctx = EvalContext(self, what, crop, self.bbox)
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

        bar = tqdm(leave=True, total=ctx.height, desc=what)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_CACHEMAX=256):
            with rasterio.open(path, "w", **meta) as dst:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    future_to_window = {
                        executor.submit(compute, win): win for win in jobs()
                    }
                    for future in concurrent.futures.as_completed(future_to_window):
                        win = future_to_window[future]
                        rows = win[0][1] - win[0][0]
                        bar.update(rows)
                        out = future.result()
                        dst.write(out.filled(meta["nodata"]), window=win, indexes=1)
        bar.close()
        return

    def open_netcdf(self, fname, **open_kwargs):
        import pdb; pdb.set_trace()
        ds = rxr.open_rasterio(fname, chunks='auto', **open_kwargs)
        if not isinstance(ds, list):
            ds = [ds]
        for group in ds:
            self.update(group)
        return

    def __repr__(self):
        return "\n".join([self[s].__repr__() for s in self.keys()])

    def tree(self, what):
        def dfs(me):
            ret = OrderedDict()
            for sym in sorted(me.inputs):
                if sym in self and self[sym].inputs:
                    ret[sym] = dfs(self[sym])
                elif sym in self and is_raster(self[sym]):
                    ret[sym] = {self[sym].__str__(): {}}
                elif sym in self and is_constant(self[sym]):
                    ret[sym] = {self[sym].is_constant: {}}
                else:
                    import pdb; pdb.set_trace()
                    ret[sym] = {}
            return ret

        deps = {what: dfs(self[what])}
        tr = asciitree.LeftAligned()
        return tr(deps)
