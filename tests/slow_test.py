import datetime
from pathlib import Path
import pytest

import fiona
import numpy as np
from r2py import modelr
from rasterset import RasterSet, Raster
import rioxarray as rxr

dir_path = Path(__file__).parent

@pytest.mark.slow
def test_dask_luh2():
    shapes = fiona.open(Path(dir_path, 's_11au16'))
    rs = RasterSet({
        'contrast_primary_vegetation_minimal_use_annual': 'c3ann + c4ann',
        'contrast_primary_vegetation_minimal_use_managed_pasture': 'pastr',
        'contrast_primary_vegetation_minimal_use_nitrogen': 'c3nfx',
        'contrast_primary_vegetation_minimal_use_perennial': 'c3per + c4per',
        'contrast_primary_vegetation_minimal_use_plantation': 0,
        'contrast_primary_vegetation_minimal_use_primary': 'primf + primn',
        'contrast_primary_vegetation_minimal_use_rangelands': 'range',
        'contrast_primary_vegetation_minimal_use_secondary': 'secdf + secdn',
        'contrast_primary_vegetation_minimal_use_timber': 0,
        'contrast_primary_vegetation_minimal_use_urban': 'urban',
        'cubrt_env_dist': 0,
        'log_adj_dist': 0,
        'hpd': 'ssp1',
        'log_hpd_rs_diff': '0 - log(hpd + 1)'
    },
                   shapes=shapes)

    rs['npp'] = Raster('/mnt/predicts/luh2/npp.tif')
    states = '/mnt/data/luh2_v2/LUH2_v2f_SSP1_RCP2.6_IMAGE/states.nc'
    luh2 = rxr.open_rasterio(states, chunks='auto', decode_times=False)[0]
    sps = rxr.open_rasterio('/mnt/predicts/luh2/sps.nc',
                            chunks='auto', decode_timedelta=True)
    sps = sps.sel(time=slice(datetime.datetime(2015, 1, 1),
                             datetime.datetime(2100, 1, 1)))
    luh2 = luh2.assign_coords(coords={'time': sps.time})
    rs.update(luh2)
    rs.update(sps)

    mod = modelr.load("/mnt/predicts/models/natgeo/cs_crop_simplemod.rds")
    rs['out'] = mod
    graph, meta = rs.build('out')
    assert meta['width'] == 1436
    assert meta['height'] == 344
    data = graph.compute()
    assert data.shape == (86, 344, 1436)
    # Need to exclude Puerto Rico and some other cell @ 158, 1292.
    assert np.allclose(data.max(), 2.300597)
    assert np.allclose(data.min(), -6.6425138)
    return