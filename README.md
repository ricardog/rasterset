# rasterset

Python library to perform calculations on sets of raster maps as if they
were one.  Each raster map is either read from disk, or computed using
"simple" expressions.  The library takes care to find the minimal
intersection of the input rasters. 

The library tries to be efficient about the calculations and will, when
possible, iterate through *block windows*.  

Uses [rasterio](https://github.com/mapbox/rasterio) to perform IO/


## Example

```python
from rasterset import RasterSet, Raster, SimpleExpr

rs = RasterSet({
	'a': Raster('/path/to/raster/a.tif'),
	'b': Raster('/path/to/raster/a.tif'),
	'two': SimpleExpr(2)
	'c': SimpleExpr('(a + log(b + 1)) * two)})

# To evaluate the whole raster
data, meta = rs.eval('c')
with rasterio.open('output.tif', 'w', **meta) as dst:
    dst.write(data, indexes=1)

# Or, to process a block at a time.  Can use multiple processes to
# process multiple blocks in parallel.
rs.write('c', 'output.tif')

```
