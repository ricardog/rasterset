from distutils.version import LooseVersion
from pathlib import Path
import warnings

import dask
from dask.array.core import normalize_chunks
import numpy as np
import numpy.ma as ma
import rasterio
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
import rasterio.windows as rwindows
from rioxarray.rioxarray import _generate_spatial_coords
from urllib.parse import urlparse
from xarray import IndexVariable
from xarray.backends.common import BackendArray
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.locks import SerializableLock
from xarray.coding import times, variables
from xarray.core import indexing
from xarray.core.dataarray import DataArray
from xarray.core.utils import is_scalar
from xarray.core.variable import as_variable

from .windows import round_window


if LooseVersion(dask.__version__) < LooseVersion("0.18.0"):
    msg = ("Automatic chunking requires dask.__version__ >= 0.18.0 . "
           "You currently have version %s" % dask.__version__
           )
    raise NotImplementedError(msg)


NETCDF_DTYPE_MAP = {
    0: object,  # NC_NAT
    1: np.byte,  # NC_BYTE
    2: np.char,  # NC_CHAR
    3: np.short,  # NC_SHORT
    4: np.int_,  # NC_INT, NC_LONG
    5: float,  # NC_FLOAT
    6: np.double,  # NC_DOUBLE
    7: np.ubyte,  # NC_UBYTE
    8: np.ushort,  # NC_USHORT
    9: np.uint,  # NC_UINT
    10: np.int64,  # NC_INT64
    11: np.uint64,  # NC_UINT64
    12: object,  # NC_STRING
}


def _to_numeric(value):
    """
    Convert the value to a number
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass
    return value


def _parse_envi(meta):
    """Parse ENVI metadata into Python data structures.

    See the link for information on the ENVI header file format:
    http://www.harrisgeospatial.com/docs/enviheaderfiles.html

    Parameters
    ----------
    meta : dict
        Dictionary of keys and str values to parse, as returned by the rasterio
        tags(ns='ENVI') call.

    Returns
    -------
    parsed_meta : dict
        Dictionary containing the original keys and the parsed values

    """

    def parsevec(value):
        return np.fromstring(value.strip("{}"), dtype="float", sep=",")

    def default(value):
        return value.strip("{}")

    parse = {"wavelength": parsevec, "fwhm": parsevec}
    parsed_meta = {key: parse.get(key, default)(value) for key, value in meta.items()}
    return parsed_meta


def _parse_tag(key, value):
    # NC_GLOBAL is appended to tags with netcdf driver and is not really
    # needed
    key = key.split("NC_GLOBAL#")[-1]
    if value.startswith("{") and value.endswith("}"):
        if value.strip("{}") == "time":
            return key, "time"
        try:
            new_val = np.fromstring(value.strip("{}"), dtype="float",
                                    sep=",")
            # pylint: disable=len-as-condition
            value = new_val if len(new_val) else _to_numeric(value)
        except ValueError:
            value = _to_numeric(value)
    else:
        value = _to_numeric(value)
    return key, value


def _parse_tags(tags):
    parsed_tags = {}
    for key, value in tags.items():
        key, value = _parse_tag(key, value)
        parsed_tags[key] = value
    return parsed_tags


def _get_rasterio_attrs(riods):
    """
    Get rasterio specific attributes
    """
    # pylint: disable=too-many-branches
    # Add rasterio attributes
    attrs = _parse_tags(riods.tags(1))
    if hasattr(riods, "nodata") and riods.nodata is not None:
        # The nodata values for the raster bands
        attrs["_FillValue"] = riods.nodata
    if hasattr(riods, "scales"):
        # The scale values for the raster bands
        if len(set(riods.scales)) > 1:
            attrs["scales"] = riods.scales
            warnings.warn(
                "Offsets differ across bands. The 'scale_factor' attribute will "
                "not be added. See the 'scales' attribute."
            )
        else:
            attrs["scale_factor"] = riods.scales[0]
    if hasattr(riods, "offsets"):
        # The offset values for the raster bands
        if len(set(riods.offsets)) > 1:
            attrs["offsets"] = riods.offsets
            warnings.warn(
                "Offsets differ across bands. The 'add_offset' attribute will "
                "not be added. See the 'offsets' attribute."
            )
        else:
            attrs["add_offset"] = riods.offsets[0]
    if hasattr(riods, "descriptions") and any(riods.descriptions):
        if len(set(riods.descriptions)) == 1:
            attrs["long_name"] = riods.descriptions[0]
        else:
            # Descriptions for each dataset band
            attrs["long_name"] = riods.descriptions
    if hasattr(riods, "units") and any(riods.units):
        # A list of units string for each dataset band
        if len(riods.units) == 1:
            attrs["units"] = riods.units[0]
        else:
            attrs["units"] = riods.units

    return attrs


def _load_netcdf_attrs(tags, data_array):
    """
    Loads the netCDF attributes into the data array

    Attributes stored in this format:
    - variable_name#attr_name: attr_value
    """
    for key, value in tags.items():
        key, value = _parse_tag(key, value)
        key_split = key.split("#")
        if len(key_split) != 2:
            continue
        variable_name, attr_name = key_split
        if variable_name in data_array.coords:
            data_array.coords[variable_name].attrs.update({attr_name: value})


def _load_netcdf_1d_coords(tags):
    """
    Dimension information:
        - NETCDF_DIM_EXTRA: '{time}' (comma separated list of dim names)
        - NETCDF_DIM_time_DEF: '{2,6}' (dim size, dim dtype)
        - NETCDF_DIM_time_VALUES: '{0,872712.659688}' (comma separated list of data)
    """
    dim_names = tags.get("NETCDF_DIM_EXTRA")
    if not dim_names:
        return {}
    dim_names = dim_names.strip("{}").split(",")
    coords = {}
    for dim_name in dim_names:
        dim_def = tags.get(f"NETCDF_DIM_{dim_name}_DEF")
        if not dim_def:
            continue
        # pylint: disable=unused-variable
        dim_size, dim_dtype = dim_def.strip("{}").split(",")
        dim_dtype = NETCDF_DTYPE_MAP.get(int(dim_dtype), object)
        dim_values = tags[f"NETCDF_DIM_{dim_name}_VALUES"].strip("{}")
        coords[dim_name] = IndexVariable(
            dim_name, np.fromstring(dim_values, dtype=dim_dtype, sep=",")
        )
    return coords


def _decode_datetime_cf(data_array, decode_times, decode_timedelta):
    """
    Decide the datetime based on CF conventions
    """
    if decode_timedelta is None:
        decode_timedelta = decode_times

    for coord in data_array.coords:
        time_var = None
        if decode_times and "since" in data_array[coord].attrs.get("units", ""):
            time_var = times.CFDatetimeCoder(use_cftime=True).decode(
                as_variable(data_array[coord]), name=coord
            )
        elif (
            decode_timedelta
            and data_array[coord].attrs.get("units") in times.TIME_UNITS
        ):
            time_var = times.CFTimedeltaCoder().decode(
                as_variable(data_array[coord]), name=coord
            )
        if time_var is not None:
            dimensions, data, attributes, encoding = variables.unpack_for_decoding(
                time_var
            )
            data_array = data_array.assign_coords(
                {
                    coord: IndexVariable(
                        dims=dimensions,
                        data=data,
                        attrs=attributes,
                        encoding=encoding,
                    )
                }
            )
    return data_array


def _parse_driver_tags(riods, attrs, coords):
    # Parse extra metadata from tags, if supported
    parsers = {"ENVI": _parse_envi}

    driver = riods.driver
    if driver in parsers:
        meta = parsers[driver](riods.tags(ns=driver))

        for key, value in meta.items():
            # Add values as coordinates if they match the band count,
            # as attributes otherwise
            if isinstance(value, (list, np.ndarray)) and len(value) == riods.count:
                coords[key] = ("band", np.asarray(value))
            else:
                attrs[key] = value


class RasterArrayWrapper(BackendArray):
    """A wrapper around rasterio dataset objects"""
    def __init__(
            self,
            manager,
            lock,
            name,
            masked
    ):
        self._manager = manager
        self._lock = lock
        self._name = name
        self._masked = masked

        # cannot save riods as an attribute: this would break pickleability
        riods = manager.acquire()
        self._shape = (riods.count, riods.height, riods.width)

        self._dtype = None
        dtypes = riods.dtypes
        if not np.all(np.asarray(dtypes) == dtypes[0]):
            raise ValueError("All bands should have the same dtype")

        self._dtype = np.dtype(dtypes[0])
        self._fill_value = riods.nodata
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

    def eval(self, key):
        return self.__getitem__(key)

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

    def _getitem(self, key):
        band_key, window, squeeze_axis, np_inds = self._get_indexer(key)
        # print(f"raster {self._name} processing window {band_key}: {window}")
        with self._lock:
            riods = self._manager.acquire()
            out = riods.read(band_key, window=window, masked=self._masked)

        if squeeze_axis:
            out = np.squeeze(out, axis=squeeze_axis)
        return out[np_inds]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )


class Raster(object):
    def __init__(self, fname, bands=None, **open_kwargs):
        self._fname = fname
        self._str = None
        self._lock = SerializableLock()

        decode_times = open_kwargs.pop('decode_times', True)
        decode_timedelta = open_kwargs.pop('decode_timedelta', None)

        with warnings.catch_warnings(record=True) as rio_warnings:
            manager = CachingFileManager(rasterio.open, self._fname,
                                         mode="r", kwargs=open_kwargs)
            riods = manager.acquire()
            captured_warnings = rio_warnings.copy()

        self._manager = manager
        riods = manager.acquire()
        if riods.subdatasets:
            raise RuntimeError("raster must be a single dataset (%s)" %
                               self._fname)
        captured_warnings = rio_warnings.copy()
        # raise the NotGeoreferencedWarning if applicable
        for rio_warning in captured_warnings:
            if not isinstance(rio_warning.message, NotGeoreferencedWarning):
                warnings.warn(str(rio_warning.message),
                              type(rio_warning.message))

        # parse tags & load alternate coords
        attrs = _get_rasterio_attrs(riods=riods)
        coords = _load_netcdf_1d_coords(riods.tags())
        _parse_driver_tags(riods=riods, attrs=attrs, coords=coords)
        for coord in coords:
            if f"NETCDF_DIM_{coord}" in attrs:
                coord_name = coord
                attrs.pop(f"NETCDF_DIM_{coord}")
                break
        else:
            coord_name = "band"
            coords[coord_name] = np.asarray(riods.indexes)
        coords.update(
            _generate_spatial_coords(riods.transform, riods.width,
                                     riods.height)
        )

        # Get bands
        if riods.count < 1:
            raise ValueError("Unknown dims")
        if bands is None:
            bands = tuple(range(1, riods.count + 1))
        elif isinstance(bands , int):
            bands = (bands, )
        self._bands = bands

        self._count = riods.count
        self._bounds = riods.bounds
        self._height = riods.height
        self._width = riods.width
        self._nodata = riods.nodata
        self._crs = riods.crs or CRS.from_epsg(4326)
        self._dtype = np.dtype(riods.dtypes[0])
        try:
            self._transform = riods.transform
        except AttributeError:
            self._transform = riods.affine

        if self.transform is None:
            raise RuntimeError("Raster %s doesn't have a geo transform" %
                               self._fname)
        assert len(set(riods.block_shapes)) == 1, "Raster has multiple block shapes"
        self._block_shape = (1,) + riods.block_shapes[0]
        self._chunks = normalize_chunks(
            chunks=(1, "auto", "auto"),
            shape=(len(self._bands), riods.height, riods.width),
            dtype=riods.dtypes[0],
            previous_chunks=tuple((c,) for c in self._block_shape),
        )
        name = attrs.pop("NETCDF_VARNAME", None)
        data = indexing.LazilyIndexedArray(
            RasterArrayWrapper(manager, lock=self._lock,
                               name=name,
                               masked=True)
        )
        xa_data = DataArray(data=data,
                            dims=(coord_name, "y", "x"), coords=coords,
                            attrs=attrs, name=name)
        _load_netcdf_attrs(riods.tags(), xa_data)
        result = _decode_datetime_cf(xa_data, decode_times=decode_times,
                                     decode_timedelta=decode_timedelta
                                     )
        self._data = result.isel({coord_name: np.array(tuple(map(lambda x: x -1,
                                                                 self._bands)))})
        self._data.set_close(manager.close)
        self._name = name
        return

    @classmethod
    def from_dataarray(cls, data):
        fname = data.encoding['source']
        if len(data.shape) != 3:
            raise RuntimeError("rioxarray Datasets should be 3-D")
        try:
            indexer = data.rio._get_obj(inplace=True)\
                              ._variable._data.array.array.key
            manager = data.rio._get_obj(inplace=True)\
                              ._variable._data.array.array.array.manager
        except AttributeError:
            raise RuntimeError("Could not determine indexer for {data.name}")
        slices = indexer.tuple
        # FIXME: Assume time/band is index 0.
        if isinstance(slices[0], np.ndarray):
            bands = slices[0]
        elif slices[0].stop is None:
            bands = np.asarray(range(data.rio.count))
        else:
            bands = np.asarray(range(slices[0].stop)[slices[0]])
        ras = Raster(fname, (bands + 1).tolist(), decode_times=False,
                     parse_coordinates=False, **manager._kwargs)
        ras.coords = data.coords
        ras.dims = data.dims
        return ras

    @property
    def syms(self):
        return []

    @property
    def inputs(self):
        return set()

    @property
    def res(self):
        return (self.transform.a, self.transform.e)

    @property
    def nodata(self):
        return self._nodata

    @property
    def bounds(self):
        return self._bounds

    @property
    def transform(self):
        return self._transform

    @property
    def crs(self):
        return self._crs

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return (len(self._bands), self._height, self._width)

    @property
    def chunks(self):
        return self._chunks

    def clip(self, bounds, inplace=False):
        if inplace:
            obj = self
        else:
            obj = self.copy()
        obj._bounds = bounds
        win = round_window(rwindows.from_bounds(*bounds,
                                                obj.transform))
        obj._transform = rwindows.transform(win, obj.transform)
        rows, cols = win.toslices()
        obj._data = obj._data.isel({'y': rows, 'x': cols}).copy()
        _, obj._height, obj._width = obj._data.shape
        obj._chunks = normalize_chunks(
            chunks=(1, "auto", "auto"),
            shape=(len(obj._bands), obj._height, obj._width),
            dtype=obj._dtype,
            previous_chunks=tuple((c,) for c in self._block_shape),
        )
        return obj

    def eval(self, df=None, win=None):
        if win is None:
            win = round_window(rwindows.from_bounds(*self.bounds,
                                                    self.transform))
        rows, cols = win.toslices()
        data = self.asarray().isel({'y': rows, 'x': cols})
        mdata = ma.masked_equal(data, self.nodata)
        return mdata

    def asarray(self):
        return self._data

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        obj = object.__new__(self.__class__)
        obj._str = None
        obj._fname = self._fname
        obj._lock = self._lock
        obj._manager = self._manager
        obj._dtype = self._dtype
        obj._bounds = self._bounds
        obj._transform = self._transform
        obj._bands = self._bands
        obj._count = self._count
        obj._width = self._height
        obj._height = self._width
        obj._nodata = self._nodata
        obj._crs = self._crs
        obj._block_shape = self._block_shape
        obj._chunks = self._chunks
        obj._name = self._name
        obj._data = self._data.copy()
        return obj

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
