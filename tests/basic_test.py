from pathlib import Path
from rasterset import Raster, RasterSet, SimpleExpr

dir_path = Path(__file__).parent

def test_sum():
    rs = RasterSet({
        'ice': Raster('ice', Path(dir_path, 'gicew-1700.tif')),
        'a': SimpleExpr('a', 1),
        'b': SimpleExpr('b', 2),
        'c': SimpleExpr('c', 'ice + a + log(b)')
    })
    data, meta = rs.eval('c')
    return


def test_mul():
    rs = RasterSet({
        'ice': Raster('ice', Path(dir_path, 'gicew-1700.tif')),
        'a': SimpleExpr('a', 1),
        'b': SimpleExpr('b', 2),
        'c': SimpleExpr('c', 'ice + a * log(b)')
    })
    data, meta = rs.eval('c')
    return
