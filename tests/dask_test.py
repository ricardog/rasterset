import pytest

import fiona
import numpy as np
import numpy.ma as ma
from pathlib import Path

from rasterset import Raster, RasterSet, SimpleExpr

dir_path = Path(__file__).parent


def test_dask():
    rs = RasterSet({'ice': Raster(Path(dir_path, 'un-codes.tif')),
                    'a': SimpleExpr(1),
                    'b': SimpleExpr(2),
                    'c': SimpleExpr('ice + a + log(b)'),
                    },
                   )
    array, meta = rs.build2('c')
    assert meta['width'] == 1440
    assert meta['height'] == 720
    data = array.load()
    if len(data.shape) == 3:
        data = data.squeeze()
    # Need to exclude Puerto Rico and some other cell @ 158, 1292.
    mdata = ma.masked_equal(data, data.attrs['_FillValue'])
    assert np.isclose(mdata.max(), 895.6932)
    assert np.isclose(mdata.min(), -97.306854)
    return


def test_dask_mask():
    shapes = fiona.open(Path(dir_path, 's_11au16'))
    rs = RasterSet({'ice': Raster(Path(dir_path, 'un-codes.tif')),
                    'a': SimpleExpr(1),
                    'b': SimpleExpr(2),
                    'c': SimpleExpr('ice + a + log(b)'),
                    },
                   shapes=shapes
                   )
    array, meta = rs.build2('c')
    assert meta['width'] == 1437
    assert meta['height'] == 344
    data = array.load()
    if len(data.shape) == 3:
        data = data.squeeze()
    # Need to exclude Puerto Rico and some other cell @ 158, 1292.
    mdata = ma.masked_equal(data, data.attrs['_FillValue'])
    assert ma.allclose(mdata[0:137, :], 840 + 1 + np.log(2))
    return
