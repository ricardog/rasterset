from pathlib import Path
from pprint import pprint
from rasterset import Raster, RasterSet, SimpleExpr

dir_path = Path(__file__).parent

def test_sum():
    rs = RasterSet({
        'ice': Raster(Path(dir_path, 'gicew-1700.tif')),
        'a': SimpleExpr(1),
        'b': SimpleExpr(2),
        'c': SimpleExpr('ice + a + log(b)')
    })
    data, meta = rs.eval('c')
    return


def test_mul():
    rs = RasterSet({
        'ice': Raster(Path(dir_path, 'gicew-1700.tif')),
        'a': SimpleExpr(1),
        'b': SimpleExpr(2),
        'c': SimpleExpr('ice + a * log(b)')
    })
    data, meta = rs.eval('c')
    return


def test_order():
    rs = RasterSet({
        'ice1': Raster(Path(dir_path, 'gicew-1700.tif')),
        'ice2': Raster(Path(dir_path, 'gicew-1700.tif')),
        'c': SimpleExpr('ice1 * ice2')
    })
    data, meta = rs.eval('c')
    return


def test_tree():
    rs = RasterSet({
        'ice': Raster(Path(dir_path, 'gicew-1700.tif')),
        'a': SimpleExpr(1),
        'b': SimpleExpr(2),
        'c': SimpleExpr('ice + a * log(b)')
    })
    pprint(rs.tree('c'))
    return
