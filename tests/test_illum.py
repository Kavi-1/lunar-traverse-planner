"""Real Site04 terrain, scalar geometry oracles, and analytical parameter tests."""

import math
from pathlib import Path

import numpy as np
import pytest
import spiceypy as spice
from rasterio import Affine
from rasterio.warp import transform

from core.cost import build_cost_surface
from core.illum import (
    FRAME,
    RADIUS_M,
    azimuth_nodes_deg,
    interpolate_horizons_deg,
    mean_shadow_fraction,
    sample_times_seconds,
    solar_angles_deg,
    spice_kernels,
    sun_positions_m,
    surface_basis,
    terrain_horizons_deg,
)
from core.terrain import compute_slope_deg, load_dem_m

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


def test_node_spacing_wrap_and_interpolation():
    # Unwrapped angles explicitly cross 360 degrees, not a 358-degree sweep.
    angles = np.array([[359.8, 360.0, 360.2], [40, 40, 40]])
    nodes = azimuth_nodes_deg(angles, 0.5)
    np.testing.assert_allclose(nodes, [[359.8, 360.2], [40, 40]])
    # Halfway between 10 and 12 degrees gives 11 degrees; constant profile is 7.
    result = interpolate_horizons_deg(nodes, [[10, 12], [7, 7]], angles)
    np.testing.assert_allclose(result, [[10, 11, 12], [7, 7, 7]])
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_horizons_deg([[0, 1]], [[0, 1]], [[2]])
    unknown = interpolate_horizons_deg([[0, 1]], [[0, np.nan]], [[0.5]])
    assert np.isnan(unknown).all()


def test_mean_shadow_hand_checked():
    # Shadow=[0,1,1], intervals 2 and 6 seconds: (1+6)/8=7/8.
    result = mean_shadow_fraction([[1, -1, -1]], [[0, 0, 0]], [0, 2, 8])
    assert result[0] == 7/8
    # Exact grazing is shadow. A single unknown epoch prevents a valid mean.
    assert mean_shadow_fraction([[0, 0]], [[0, 0]], [0, 1])[0] == 1
    assert np.isnan(mean_shadow_fraction([[0, 0]], [[0, np.nan]], [0, 1])[0])
    with pytest.raises(ValueError):
        mean_shadow_fraction([[0, 0]], [[0, 0]], [1, 0])


def _scalar_profile_deg(z_m, t_m, row, col, azimuth_deg, radius_m=RADIUS_M):
    """Independent scalar Cartesian oracle over an actual small DEM patch.

Use lat/lon trigonometry and direct Cartesian subtraction, instead of the
production rational inverse and stable radial/horizontal difference formula.
A 1 mm initial probe independently approximates the one-sided tangent.
"""
    x0, y0 = t_m.c+(col+.5)*t_m.a, t_m.f+(row+.5)*t_m.e
    lat = -math.pi/2+2*math.atan(math.hypot(x0, y0)/(2*RADIUS_M))
    lon = math.atan2(x0, y0)
    up = np.array([math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0])
    north = np.array([-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)])
    heading = north*math.cos(math.radians(azimuth_deg))+east*math.sin(math.radians(azimuth_deg))
    observer = (radius_m+z_m[row, col])*up
    maximum = -90.0
    for distance in [0.001, *np.arange(5, 200, 5)]:
        q = up*math.cos(distance/RADIUS_M)+heading*math.sin(distance/RADIUS_M)
        lon_q, lat_q = math.atan2(q[1], q[0]), math.asin(q[2])
        rho = 2*RADIUS_M*math.tan((lat_q+math.pi/2)/2)
        x, y = rho*math.sin(lon_q), rho*math.cos(lon_q)
        r, c = (y-t_m.f)/t_m.e-.5, (x-t_m.c)/t_m.a-.5
        if not (0 <= r < len(z_m)-1 and 0 <= c < z_m.shape[1]-1):
            break
        ri, ci = math.floor(r), math.floor(c)
        h = sum(z_m[ri+dr, ci+dc]*wr*wc
                for dr, wr in ((0, 1-(r-ri)), (1, r-ri))
                for dc, wc in ((0, 1-(c-ci)), (1, c-ci)))
        delta = (radius_m+h)*q-observer
        vertical = float(delta @ up)
        angle = math.degrees(math.atan2(vertical, np.linalg.norm(delta-vertical*up)))
        maximum = max(maximum, angle)
    return maximum


@pytest.mark.parametrize("azimuth_deg", [0, 40, 90, 180, 270])
def test_horizon_against_scalar_real_patch(dem, azimuth_deg):
    z, t, _ = dem
    patch = z[2196:2205, 776:785]
    patch_t = t @ Affine.translation(776, 2196)
    result = terrain_horizons_deg(patch, patch_t, [[4, 4]], [[azimuth_deg]])
    reference = _scalar_profile_deg(patch, patch_t, 4, 4, azimuth_deg)
    # The scalar 1 mm probe approximates the tangent, and direct subtraction
    # loses more precision than production. 0.002 deg is an oracle tolerance.
    assert result["horizon_deg"][0, 0] == pytest.approx(reference, abs=0.002)
    if azimuth_deg == 0:
        # Independently checked meridian profile at d=20 m: interpolated
        # z=1259.725376332149 m versus observer z=1261.243896484375 m.
        # Cartesian vertical=-1.51863535027951 m, horizontal=20.014501270150028 m.
        # degrees(atan2(vertical,horizontal))=-4.339103255372535 degrees.
        assert result["horizon_deg"][0, 0] == pytest.approx(-4.339103255372535, abs=3e-8)
    assert result["sample_count"] > 0
    assert 0 < result["boundary_distance_m"][0, 0] < 50


def test_horizon_unknown_and_edge_validation(dem):
    z, t, _ = dem
    patch = np.ma.array(z[2196:2205, 776:785], mask=False)
    patch.mask[4, 4] = True
    result = terrain_horizons_deg(patch, t @ Affine.translation(776, 2196), [[4, 4]], [[40]])
    assert np.isnan(result["horizon_deg"]).all()
    with pytest.raises(ValueError, match="interior"):
        terrain_horizons_deg(z, t, [[0, 0]], [[0]])
    with pytest.raises(ValueError):
        terrain_horizons_deg(z, t, [[4, 4]], [[0]], distance_step_m=0)


def test_radius_sensitivity_holds_geographic_samples_fixed(dem):
    z, t, _ = dem
    patch = z[2196:2205, 776:785]
    patch_t = t @ Affine.translation(776, 2196)
    nominal = terrain_horizons_deg(patch, patch_t, [[4, 4]], [[40]])
    for radius_m in (RADIUS_M-1000, RADIUS_M+1000):
        variant = terrain_horizons_deg(patch, patch_t, [[4, 4]], [[40]], radius_m=radius_m)
        expected = _scalar_profile_deg(patch, patch_t, 4, 4, 40, radius_m)
        assert variant["horizon_deg"][0, 0] == pytest.approx(expected, abs=0.002)
        np.testing.assert_array_equal(variant["boundary_distance_m"], nominal["boundary_distance_m"])
        assert variant["sample_count"] == nominal["sample_count"]


def test_real_near_field_peak_explains_spatial_sampling_difference(dem):
    z, t, _ = dem
    cell, azimuth = [[2086, 608]], [[32.52282386568365]]
    coarse = terrain_horizons_deg(z, t, cell, azimuth, distance_step_m=5)
    fine = terrain_horizons_deg(z, t, cell, azimuth, distance_step_m=2.5)
    # Independent scalar Cartesian arithmetic, recorded in DECISIONS.md:
    # observer z=1357.462646484375 m; z(22.5 m)=1362.485662927584 m,
    # z(25 m)=1361.4077937288582 m. Geographic sample locations, bilinear
    # weights and direct Cartesian differences give these angles. This tests a
    # known sampling limitation, not a claimed continuous-horizon ground truth.
    assert coarse["horizon_deg"][0, 0] == pytest.approx(8.960350105496337, abs=1e-7)
    assert fine["horizon_deg"][0, 0] == pytest.approx(12.574747402501846, abs=1e-7)
    assert coarse["horizon_distance_m"][0, 0] == 25
    assert fine["horizon_distance_m"][0, 0] == 22.5


def test_shadow_cost_on_real_slope(dem):
    z, t, _ = dem
    slope = compute_slope_deg(z[999:1002, 999:1002], t)
    baseline = build_cost_surface(slope, slope_limit_deg=20, slope_weight=2)
    # A prescribed half-window shadow with weight 2 adds exactly 1 weighted
    # meter per projected meter. This is a cost parameter test, not a fake DEM.
    fraction = np.full(slope.shape, 0.5)
    weighted = build_cost_surface(slope, slope_limit_deg=20, slope_weight=2,
                                  shadow_fraction=fraction, shadow_weight=2)
    assert weighted[1, 1] == pytest.approx(baseline[1, 1]+1, abs=1e-14)
    assert np.array_equal(np.isinf(weighted), np.isinf(baseline))
    fraction[1, 1] = np.nan
    assert np.isinf(build_cost_surface(slope, slope_limit_deg=20, slope_weight=2,
                                      shadow_fraction=fraction, shadow_weight=2)).all()
    np.testing.assert_array_equal(build_cost_surface(
        slope, slope_limit_deg=20, slope_weight=2, shadow_fraction=fraction, shadow_weight=0), baseline)
    for bad in (None, [[0.5]], np.full(slope.shape, 1.1)):
        with pytest.raises(ValueError):
            build_cost_surface(slope, slope_limit_deg=20, slope_weight=2,
                               shadow_fraction=bad, shadow_weight=2)
    with pytest.raises(ValueError):
        build_cost_surface(slope, slope_limit_deg=20, slope_weight=2, shadow_weight=-1)
