from pathlib import Path
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
