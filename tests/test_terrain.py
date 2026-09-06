"""Tests use actual Site04 elevations; no synthetic terrain fixtures."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from core.terrain import load_dem_m

DEM_PATH = Path(__file__).resolve().parents[1] / "data/Site04/Site04_final_adj_5mpp_surf.tif"


@pytest.fixture(scope="module")
def dem():
    return load_dem_m(DEM_PATH)


def test_load_real_dem_m(dem):
    elevation_m, transform_m, crs = dem
    assert elevation_m.shape == (3200, 3200)
    assert elevation_m.dtype == np.float64
    assert elevation_m[1000, 1000] == 1235.669921875
    assert transform_m == Affine(5, 0, -9000, 0, -5, 1000)
    # GeoTIFF affine locates the corner; PGDA samples lie at pixel centers.
    assert rasterio.transform.xy(transform_m, 0, 0) == (-8997.5, 997.5)
    with rasterio.open(DEM_PATH) as source:
        assert crs == source.crs
        assert np.array_equal(np.isnan(elevation_m), source.read_masks(1) == 0)


def test_missing_dem_stops(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_site04.py"):
        load_dem_m(tmp_path / "missing.tif")
