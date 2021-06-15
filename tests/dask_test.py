import fiona
import numpy as np
from pathlib import Path

from rasterset import Raster, RasterSet, SimpleExpr

dir_path = Path(__file__).parent

def test_dask():
    shapes = fiona.open(Path(dir_path, 's_11au16'))
    rs = RasterSet({'ice': Raster(Path(dir_path, 'un-codes.tif')),
                    'a': SimpleExpr(1),
                    'b': SimpleExpr(2),
                    'c': SimpleExpr('ice + a + log(b)'),
                    },
                   shapes=shapes
                   )
    graph, meta = rs.build('c')
    assert meta['width'] == 1436
    assert meta['height'] == 344
    data = graph.compute()
    if len(data.shape) == 3:
        data = data.squeeze()
    # Need to exclude Puerto Rico and some other cell @ 158, 1292.
    assert np.allclose(data[0:137, :], 840 + 1 + np.log(2))
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
    graph, meta = rs.build('c')
    assert meta['width'] == 1436
    assert meta['height'] == 344
    data = graph.compute()
    if len(data.shape) == 3:
        data = data.squeeze()
    # Need to exclude Puerto Rico and some other cell @ 158, 1292.
    assert np.allclose(data[0:137, :], 840 + 1 + np.log(2))
    return
