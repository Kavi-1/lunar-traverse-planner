"""Tests use actual Site04 elevations; no synthetic terrain fixtures."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from core.terrain import compute_slope_deg, load_dem_m

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


@pytest.mark.parametrize(
    "row,column,expected_deg",
    [
        (1000, 1000, 17.79828911404621),
        (1600, 1600, 32.84041803196029),
        (500, 500, 8.397770712777001),
    ],
)
def test_horn_real_neighborhood_hand_checked(dem, row, column, expected_deg):
    elevation_m, transform_m, _ = dem
    neighborhood_m = elevation_m[row - 1 : row + 2, column - 1 : column + 2]
    # Independent scalar arithmetic on the downloaded float32 values:
    # (1000,1000): weighted X,Y numerators = 7.88134765625, 10.13818359375 m.
    # Divide by +40,-40 m -> gradients 0.19703369140625,-0.25345458984375.
    # (1600,1600): -25.717041015625,-2.283203125 m ->
    #             -0.642926025390625,0.057080078125.
    # (500,500): 3.979705810546875,4.362579345703125 m ->
    #           0.099492645263671875,-0.109064483642578125.
    # Each expected angle is atan(sqrt(gx*gx + gy*gy)) * 180/pi.
    result_deg = compute_slope_deg(neighborhood_m, transform_m, method="horn")
    assert result_deg[1, 1] == pytest.approx(expected_deg, abs=1e-12)
    assert np.count_nonzero(np.isfinite(result_deg)) == 1


@pytest.mark.parametrize("method", ["central", "horn", "forward"])
@pytest.mark.parametrize("row_offset,column_offset", [(r, c) for r in range(3) for c in range(3)])
def test_invalid_neighbor_excludes_center(dem, row_offset, column_offset, method):
    elevation_m, transform_m, _ = dem
    # Mask observations in a real neighborhood to test missing-data handling;
    # do not generate or replace any terrain heights.
    neighborhood_m = np.ma.array(elevation_m[999:1002, 999:1002], mask=False)
    neighborhood_m.mask[row_offset, column_offset] = True
    assert np.isnan(compute_slope_deg(neighborhood_m, transform_m, method=method)).all()


def test_unsupported_grid_stops(dem):
    elevation_m, transform_m, _ = dem
    patch_m = elevation_m[999:1002, 999:1002]
    with pytest.raises(ValueError, match="Rotated or sheared"):
        compute_slope_deg(patch_m, Affine(5, 1, -9000, 0, -5, 1000))
    with pytest.raises(ValueError, match="nonzero"):
        compute_slope_deg(patch_m, Affine(0, 0, 0, 0, -5, 0))
    with pytest.raises(ValueError, match="at least 3"):
        compute_slope_deg(patch_m[:2], transform_m)


@pytest.mark.parametrize(
    "method,expected_deg",
    [("central", 17.646201418369948), ("forward", 17.882469994042452)],
)
def test_unweighted_real_neighborhood_hand_checked(dem, method, expected_deg):
    elevation_m, transform_m, _ = dem
    # Actual (1000,1000) neighborhood: central X,Y differences are
    # 1.95751953125, 2.5074462890625 m, over +10,-10 m respectively.
    # Forward differences are 0.946044921875,1.3067626953125 m over +5,-5 m.
    # Expected degrees = atan(sqrt(gx*gx + gy*gy)) * 180/pi.
    result_deg = compute_slope_deg(elevation_m[999:1002, 999:1002], transform_m, method=method)
    assert result_deg[1, 1] == pytest.approx(expected_deg, abs=1e-12)
    assert np.count_nonzero(np.isfinite(result_deg)) == 1


def test_production_default_is_central(dem):
    elevation_m, transform_m, _ = dem
    result_deg = compute_slope_deg(elevation_m[999:1002, 999:1002], transform_m)
    assert result_deg[1, 1] == pytest.approx(17.646201418369948, abs=1e-12)
    with pytest.raises(ValueError, match="Unknown slope method"):
        compute_slope_deg(elevation_m[999:1002, 999:1002], transform_m, method="unknown")


def test_clone_elevations_and_slope_hand_checked(dem):
    from core.clones import terrain_window_m

    clone_path = DEM_PATH.parent / 'Clones/Site04_final_adj_5mpp_0001_err.tif'
    elevation_m, slope_deg, _ = terrain_window_m(clone_path)
    base_m, _, _ = terrain_window_m(DEM_PATH)
    # Actual downloaded float32 samples, not generated terrain. Window cell
    # (200,80) is source (2200,660). Clone is already an elevation in meters.
    assert elevation_m[200, 80] == 1435.2269287109375
    assert elevation_m[200, 80] - base_m[200, 80] == -0.0467529296875
    assert elevation_m[200, 200] - base_m[200, 200] == 0.514404296875
    # Opposite neighbors: X=(1435.7071533203125-1434.6396484375)/10;
    # Y=(1434.6376953125-1435.956787109375)/-10. atan(hypot(X,Y)) in deg.
    assert slope_deg[200, 80] == pytest.approx(9.630946116730064, abs=1e-12)
    # Center: X=(1262.10791015625-1261.85009765625)/10;
    # Y=(1260.650634765625-1262.8253173828125)/-10.
    assert slope_deg[200, 200] == pytest.approx(12.352271767161753, abs=1e-12)


def test_analysis_halo_matches_full_dem(dem):
    from core.clones import terrain_window_m

    full_m, transform_m, _ = dem
    cropped_m, slope_deg, cropped_transform_m = terrain_window_m(DEM_PATH)
    np.testing.assert_array_equal(cropped_m, full_m[2000:2400, 580:980])
    np.testing.assert_array_equal(slope_deg,
                                 compute_slope_deg(full_m, transform_m)[2000:2400, 580:980])
    assert cropped_transform_m == Affine(5, 0, -6100, 0, -5, -9000)


def test_clone_hash_validation(tmp_path):
    from core.clones import input_hash

    path = tmp_path / 'input'
    path.write_bytes(b'abc')
    path.with_suffix('.sha256').write_text(
        'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n')
    assert input_hash(path).startswith('ba7816bf')
    path.write_bytes(b'abcd')
    with pytest.raises(ValueError, match='Hash mismatch'):
        input_hash(path)


def test_clone_download_selection(tmp_path, monkeypatch):
    from scripts import download_site04

    requested = []
    monkeypatch.setattr(download_site04, 'download_file',
                        lambda url, **kwargs: requested.append((url, kwargs['output_dir'])))
    download_site04.download_site04(tmp_path, clones_only=True)
    assert len(requested) == 100
    assert requested[0][0].endswith('/Clones/Site04_final_adj_5mpp_0001_err.tif')
    assert requested[-1][0].endswith('/Clones/Site04_final_adj_5mpp_0100_err.tif')
    assert all(directory == tmp_path / 'Clones' for _, directory in requested)
    with pytest.raises(ValueError):
        download_site04.download_site04(tmp_path, clones_only=True, kernels_only=True)


def test_download_preserves_real_dem_bytes(tmp_path):
    from core.clones import input_hash
    from scripts.download_site04 import download_file

    # Local file URL exercises the exact copy/hash/reuse path without a network
    # request or invented terrain. Corrupt only the test copy to verify rejection.
    download_file(DEM_PATH.as_uri(), tmp_path, 60, False)
    downloaded = tmp_path / DEM_PATH.name
    assert input_hash(downloaded) == input_hash(DEM_PATH)
    download_file(DEM_PATH.as_uri(), tmp_path, 60, False)
    with downloaded.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(RuntimeError, match='Hash mismatch'):
        download_file(DEM_PATH.as_uri(), tmp_path, 60, False)
    assert not downloaded.with_suffix('.tif.part').exists()


def test_clone_preflight_missing_inputs_stops(tmp_path):
    from core.clones import run_site04

    with pytest.raises(FileNotFoundError, match='Missing'):
        run_site04(tmp_path)
    assert not (tmp_path / 'figure.png').exists()


def test_clone_projection_matches_despite_wkt_names():
    from rasterio.crs import CRS

    from core.clones import matching_projection

    clone_path = DEM_PATH.parent / 'Clones/Site04_final_adj_5mpp_0001_err.tif'
    with rasterio.open(DEM_PATH) as nominal, rasterio.open(clone_path) as clone:
        assert nominal.crs.to_wkt() != clone.crs.to_wkt()
        assert matching_projection(clone.crs, nominal.crs)
        altered = clone.crs.to_dict()
        altered['R'] += 1  # Deliberately malformed registration, not a physical assumption.
        assert not matching_projection(CRS.from_dict(altered), nominal.crs)
        assert not matching_projection(None, nominal.crs)
