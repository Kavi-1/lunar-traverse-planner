"""Real Site04 terrain, scalar geometry oracles, and analytical parameter tests."""

from pathlib import Path

import numpy as np
import pytest
import spiceypy as spice
from rasterio.warp import transform

from core.illum import (
    FRAME,
    RADIUS_M,
    sample_times_seconds,
    solar_angles_deg,
    spice_kernels,
    sun_positions_m,
    surface_basis,
)
from core.terrain import load_dem_m

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dem():
    return load_dem_m(ROOT / "data/Site04/Site04_final_adj_5mpp_surf.tif")


def test_basis_hand_checked_and_inverse_projection(dem):
    # At the south pole, longitude zero convention gives up=-Z,east=Y,north=X.
    up, east, north = surface_basis([0], [0])
    np.testing.assert_allclose(up, [[0, 0, -1]], atol=1e-15)
    np.testing.assert_allclose(east, [[0, 1, 0]], atol=1e-15)
    np.testing.assert_allclose(north, [[1, 0, 0]], atol=1e-15)
    x_m, y_m = [-5097.5, -8997.5], [-10002.5, 997.5]
    up, east, north = surface_basis(x_m, y_m)
    lon_deg, lat_deg = transform(dem[2], {"proj": "longlat", "R": RADIUS_M}, x_m, y_m)
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    reference = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    np.testing.assert_allclose(up, reference, atol=1e-13)
    np.testing.assert_allclose(np.cross(east, north), up, atol=1e-14)
    np.testing.assert_allclose(np.sum(up*north, axis=1), 0, atol=1e-14)


def test_solar_angle_convention_at_real_observer(dem):
    z_m = dem[0][2200, 780]
    up, east, north = surface_basis([-5097.5], [-10002.5])
    position_m = up*(RADIUS_M+z_m)
    # Analytical direction vectors, not fabricated terrain: equal north/up
    # components imply elevation 45 degrees; pure east implies azimuth 90.
    directions = np.vstack((position_m+1000*(north+up), position_m+1000*east))
    az, el = solar_angles_deg([-5097.5], [-10002.5], [z_m], directions)
    np.testing.assert_allclose(az, [[0, 90]], atol=1e-10)
    np.testing.assert_allclose(el, [[45, 0]], atol=1e-10)


def test_kernels_time_and_independent_frame_transform():
    with spice_kernels(ROOT / "data/kernels"):
        times = sample_times_seconds("2026-09-06T00:00:00", "2026-09-06T00:10:00", 240)
        # Actual ET intervals: four minutes, four minutes, then two minutes.
        np.testing.assert_allclose(np.diff(times), [240, 240, 120], atol=1e-5)
        sun = sun_positions_m(times)
        for t, direct in zip(times, sun):
            inertial = spice.spkpos("SUN", float(t), "J2000", "CN+S", "MOON")[0]*1000
            rotation = spice.pxform("J2000", FRAME, float(t))
            np.testing.assert_allclose(rotation @ inertial, direct, rtol=1e-14, atol=1e-3)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
            assert np.linalg.det(rotation) == pytest.approx(1, abs=1e-14)
        with pytest.raises(spice.utils.exceptions.SpiceyError):
            sun_positions_m([spice.str2et("2100-01-01")])
        with pytest.raises(ValueError):
            sample_times_seconds("2026-09-06", "2026-09-05", 300)
        with pytest.raises(ValueError, match="UTC"):
            sample_times_seconds("2026-09-06T00:00:00-04:00", "2026-09-07", 300)
        np.testing.assert_allclose(times, sample_times_seconds(
            "2026-09-06T00:00:00Z", "2026-09-06T00:10:00+00:00", 240), atol=1e-9)
        # Nested use must leave outer ownership intact.
        with spice_kernels(ROOT / "data/kernels"):
            sun_positions_m(times[:1])
        sun_positions_m(times[:1])


def test_missing_kernel_stops(tmp_path):
    with pytest.raises(FileNotFoundError, match="kernels-only"), spice_kernels(tmp_path):
        pass
