"""Cost and routing checks use actual Site04 observations, never synthetic terrain."""

import math
from pathlib import Path

import numpy as np
import pytest
from rasterio import Affine

from core.cost import build_cost_surface
from core.plan import find_route, route_statistics
from core.terrain import compute_slope_deg, load_dem_m

DEM_PATH = Path(__file__).resolve().parents[1] / "data/Site04/Site04_final_adj_5mpp_surf.tif"


@pytest.fixture(scope="module")
def slope_patch_deg():
    elevation_m, transform_m, _ = load_dem_m(DEM_PATH)
    return compute_slope_deg(elevation_m[999:1005, 999:1005], transform_m)


def test_cost_real_sample_hand_checked(slope_patch_deg):
    weights = build_cost_surface(slope_patch_deg, slope_limit_deg=25, slope_weight=2)
    # Independent derivation from raw DEM neighbors of (1000,1000), in meters:
    # gx = (1236.615966796875 - 1234.658447265625)/10 = 4009/20480.
    # gy = (1236.9766845703125 - 1234.46923828125)/-10 = -20541/81920.
    # Signed two-pixel spacings are +10 m in X and -10 m in Y.
    # Exact rational arithmetic, bypassing the production atan/tan path:
    # 1 + 2*(gx*gx + gy*gy) = 4034529177/3355443200
    #                     = 1.202383392155170440673828125.
    # The original literal 1.2023771303892135 was entered incorrectly;
    # no derivation supports it, and rounding was not established as its cause.
    assert weights[1, 1] == pytest.approx(1.2023833921551703, abs=1e-12)
    assert np.isinf(weights[0]).all()
    assert weights.dtype == np.float64


def test_cutoff_equality_and_zero_preference(slope_patch_deg):
    cutoff_deg = float(slope_patch_deg[1, 1])
    original_deg = slope_patch_deg.copy()
    weights = build_cost_surface(slope_patch_deg, slope_limit_deg=cutoff_deg, slope_weight=0)
    allowed = np.isfinite(slope_patch_deg) & (slope_patch_deg <= cutoff_deg)
    assert np.all(weights[allowed] == 1)
    assert np.isinf(weights[~allowed]).all()
    assert weights[1, 1] == 1
    lower_weights = build_cost_surface(
        slope_patch_deg, slope_limit_deg=np.nextafter(cutoff_deg, 0), slope_weight=0
    )
    assert np.isinf(lower_weights[1, 1])
    np.testing.assert_array_equal(slope_patch_deg, original_deg)


def test_masked_observation_is_blocked(slope_patch_deg):
    masked_deg = np.ma.array(slope_patch_deg, mask=False)
    masked_deg.mask[1, 1] = True
    weights = build_cost_surface(masked_deg, slope_limit_deg=25, slope_weight=2)
    assert np.isinf(weights[1, 1])


@pytest.mark.parametrize("slope_limit_deg", [0, -1, 90, np.nan, np.inf])
def test_invalid_cutoff(slope_patch_deg, slope_limit_deg):
    with pytest.raises(ValueError, match="slope_limit_deg"):
        build_cost_surface(slope_patch_deg, slope_limit_deg=slope_limit_deg, slope_weight=1)


@pytest.mark.parametrize("slope_weight", [-1, np.nan, np.inf])
def test_invalid_preference(slope_patch_deg, slope_weight):
    with pytest.raises(ValueError, match="slope_weight"):
        build_cost_surface(slope_patch_deg, slope_limit_deg=25, slope_weight=slope_weight)


def test_invalid_array_shape(slope_patch_deg):
    for values_deg in (slope_patch_deg[1], slope_patch_deg[:0]):
        with pytest.raises(ValueError, match="nonempty 2D"):
            build_cost_surface(values_deg, slope_limit_deg=25, slope_weight=1)


def test_malformed_negative_slope_rejected(slope_patch_deg):
    # Negate actual observations solely to exercise invalid input handling.
    with pytest.raises(ValueError, match="Finite slope_deg"):
        build_cost_surface(-slope_patch_deg, slope_limit_deg=25, slope_weight=1)


@pytest.fixture(scope="module")
def routing_patch(slope_patch_deg):
    # Three by three actual slope samples, with the original pixel registration.
    # Computing slopes before slicing preserves all source neighborhoods.
    weights = build_cost_surface(slope_patch_deg[1:4, 1:4], slope_limit_deg=25, slope_weight=2)
    assert np.isfinite(weights).all()
    return weights, Affine(5, 0, -4000, 0, -5, -4000)


def enumerate_minimum_cost(weights, spacing_y_m, spacing_x_m):
    """Test-only exhaustive simple paths; explicitly capped at 4x4 real samples."""
    assert weights.ndim == 2 and max(weights.shape) <= 4
    goal = (weights.shape[0] - 1, weights.shape[1] - 1)
    best_cost = math.inf

    def visit(cell, visited, cost):
        nonlocal best_cost
        if cell == goal:
            best_cost = min(best_cost, cost)
            return
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (cell[0] + dr, cell[1] + dc)
                if neighbor in visited:
                    continue
                r, c = neighbor
                if not (0 <= r < weights.shape[0] and 0 <= c < weights.shape[1]):
                    continue
                if not math.isfinite(weights[neighbor]):
                    continue
                distance_m = math.sqrt((dr * spacing_y_m)**2 + (dc * spacing_x_m)**2)
                edge_cost = distance_m * (float(weights[cell]) + float(weights[neighbor])) / 2
                visit(neighbor, visited | {neighbor}, cost + edge_cost)

    visit((0, 0), {(0, 0)}, 0.0)
    return best_cost


def test_route_matches_exhaustive_real_patch(routing_patch):
    weights, transform_m = routing_patch
    result = find_route(weights, transform_m, (-3997.5, -4002.5), (-3987.5, -4012.5))
    expected_cost = enumerate_minimum_cost(weights, 5, 5)
    assert result["total_cost_weighted_m"] == pytest.approx(expected_cost, abs=1e-12)
    route = result["route_rc"]
    np.testing.assert_array_equal(route[0], (0, 0))
    np.testing.assert_array_equal(route[-1], (2, 2))
    assert len(set(map(tuple, route))) == len(route)
    assert np.all(np.max(np.abs(np.diff(route, axis=0)), axis=1) == 1)
    assert np.isfinite(weights[route[:, 0], route[:, 1]]).all()
    reverse = find_route(weights, transform_m, (-3987.5, -4012.5), (-3997.5, -4002.5))
    assert reverse["total_cost_weighted_m"] == pytest.approx(expected_cost, abs=1e-12)


def test_route_geometry_hand_checked(slope_patch_deg):
    weights = build_cost_surface(slope_patch_deg[1:4, 1:4], slope_limit_deg=25, slope_weight=0)
    # Same real slope observations; deliberately unequal affine spacings test
    # sampling order, not a claim about the source DEM's georeferencing.
    transform_m = Affine(3, 0, 0, 0, -4, 0)
    result = find_route(weights, transform_m, (1, -1), (4.5, -10))
    # Centers are (1.5,-2) and (4.5,-10). One 3-4-5 diagonal plus one
    # vertical 4 m step costs 9 weighted m at zero slope preference.
    # Start snap is sqrt(0.5²+(-1)²) = sqrt(1.25); goal snap is zero.
    assert result["total_cost_weighted_m"] == pytest.approx(9)
    assert result["snapped_start_xy_m"] == (1.5, -2)
    assert result["start_snap_distance_m"] == pytest.approx(math.sqrt(1.25))
    assert result["goal_snap_distance_m"] == 0
    assert result["diagonal_blocked_side_steps"] == 0
    np.testing.assert_array_equal(result["route_xy_m"][-1], (4.5, -10))


@pytest.mark.parametrize("both_blocked", [False, True])
def test_diagonal_corner_contacts(routing_patch, both_blocked):
    weights, transform_m = routing_patch
    masked_weights = np.ma.array(weights[:2, :2], mask=False)
    masked_weights.mask[0, 1] = True
    masked_weights.mask[1, 0] = both_blocked
    result = find_route(masked_weights, transform_m, (-3997.5, -4002.5), (-3992.5, -4007.5))
    assert result["status"] == "ok"
    # The direct diagonal contacts side cells (0,1) and (1,0). Mask one or
    # both actual samples: exactly one contact step, with the stated subset.
    assert result["diagonal_blocked_side_steps"] == 1
    assert result["diagonal_both_blocked_steps"] == int(both_blocked)
    assert len(result["route_rc"]) == 2


def test_disconnected_real_patch(routing_patch):
    weights, transform_m = routing_patch
    masked_weights = np.ma.array(weights, mask=False)
    masked_weights.mask[1, :] = True
    result = find_route(masked_weights, transform_m, (-3997.5, -4002.5), (-3987.5, -4012.5))
    assert result["status"] == "no_route"
    assert result["route_rc"] is None
    assert result["total_cost_weighted_m"] is None
    assert result["diagonal_blocked_side_steps"] is None


def test_same_cell(routing_patch):
    weights, transform_m = routing_patch
    result = find_route(weights, transform_m, (-3999, -4001), (-3997, -4003))
    assert result["status"] == "ok"
    assert result["total_cost_weighted_m"] == 0
    assert result["diagonal_blocked_side_steps"] == 0
    np.testing.assert_array_equal(result["route_rc"], [[0, 0]])


@pytest.mark.parametrize("point_xy_m,match", [
    ((-4000.1, -4002.5), "outside"),
    ((-3985, -4002.5), "outside"),
    ((-3997.5, -4015), "outside"),
    ((np.nan, -4002.5), "finite projected"),
])
def test_invalid_endpoint(routing_patch, point_xy_m, match):
    weights, transform_m = routing_patch
    with pytest.raises(ValueError, match=match):
        find_route(weights, transform_m, point_xy_m, (-3987.5, -4012.5))


def test_blocked_endpoint(routing_patch):
    weights, transform_m = routing_patch
    masked_weights = np.ma.array(weights, mask=False)
    masked_weights.mask[0, 0] = True
    with pytest.raises(ValueError, match="start is blocked"):
        find_route(masked_weights, transform_m, (-3997.5, -4002.5), (-3987.5, -4012.5))


@pytest.mark.parametrize("transform_m", [
    Affine(5, 1, 0, 0, -5, 0), Affine(0, 0, 0, 0, -5, 0),
    Affine(5, 0, np.nan, 0, -5, 0),
])
def test_invalid_routing_transform(routing_patch, transform_m):
    weights, _ = routing_patch
    with pytest.raises(ValueError):
        find_route(weights, transform_m, (-3997.5, -4002.5), (-3987.5, -4012.5))


def test_invalid_routing_weights(routing_patch):
    weights, transform_m = routing_patch
    for invalid in (-weights, weights[0], weights[:0]):
        with pytest.raises(ValueError, match="cost_weights"):
            find_route(invalid, transform_m, (-3997.5, -4002.5), (-3987.5, -4012.5))


def test_route_parameter_monotonicity(slope_patch_deg, routing_patch):
    _, transform_m = routing_patch
    slopes_deg = slope_patch_deg[1:4, 1:4]
    start_xy_m, goal_xy_m = (-3997.5, -4002.5), (-3987.5, -4012.5)
    costs = []
    for slope_weight in (0, 2, 4):
        weights = build_cost_surface(slopes_deg, slope_limit_deg=25, slope_weight=slope_weight)
        costs.append(find_route(weights, transform_m, start_xy_m, goal_xy_m)[
            "total_cost_weighted_m"
        ])
    assert costs[0] <= costs[1] <= costs[2]
    # Use actual endpoint slopes to keep both endpoints valid. Relaxing only
    # the cutoff adds allowed cells, so optimum cost cannot increase.
    cutoff_deg = float(max(slopes_deg[0, 0], slopes_deg[-1, -1]))
    restricted_weights = build_cost_surface(slopes_deg, slope_limit_deg=cutoff_deg, slope_weight=2)
    restricted = find_route(restricted_weights, transform_m, start_xy_m, goal_xy_m)
    if restricted["status"] == "ok":
        assert costs[1] <= restricted["total_cost_weighted_m"] + 1e-12
    else:
        assert restricted["total_cost_weighted_m"] is None


@pytest.fixture(scope="module")
def statistics_patch():
    elevation_m, transform_m, _ = load_dem_m(DEM_PATH)
    patch_m = elevation_m[999:1004, 999:1004]
    return patch_m, compute_slope_deg(patch_m, transform_m), transform_m


def test_statistics_hand_checked(statistics_patch):
    heights_m, slopes_deg, transform_m = statistics_patch
    result = route_statistics([[1, 1], [1, 2], [2, 3]], heights_m, slopes_deg, transform_m)
    # Independent scalar arithmetic from the raw DEM, without production helpers:
    # heights = 1235.669921875, 1236.615966796875, 1239.0230712890625 m.
    # rises = 0.946044921875, 2.4071044921875 m; runs = 5, sqrt(50) m.
    # length = 5+sqrt(50) = 12.071067811865476 m.
    # polyline = sqrt(25+0.946044921875²)+sqrt(50+2.4071044921875²)
    #          = 12.558261413458432 m.
    # Ascent/net = 0.946044921875+2.4071044921875 = 3.3531494140625 m.
    # No negative rise means zero descent.
    # Central gx,gy from opposite raw neighbors, divided by +10,-10 m:
    # (1236.615966796875-1234.658447265625)/10 = 0.195751953125;
    # (1236.9766845703125-1234.46923828125)/-10 = -0.25074462890625.
    # (1237.59619140625-1235.669921875)/10 = 0.192626953125;
    # (1238.0115966796875-1235.3369140625)/-10 = -0.26746826171875.
    # (1240.208251953125-1238.0115966796875)/10 = 0.21966552734375;
    # (1240.349365234375-1237.59619140625)/-10 = -0.2753173828125.
    # atan(sqrt(gx²+gy²))*180/pi gives slopes s0,s1,s2:
    # 17.646201418369948, 18.242866980426832, 19.402825626556908 deg.
    # Mean = [5*(s0+s1)/2+sqrt(50)*(s1+s2)/2]/[5+sqrt(50)]
    #      = 18.459037517979905 deg; max is s2.
    # Grades = atan(rise/run)*180/pi = 10.714218007014956,
    # 18.799394807511753 deg; the latter is max absolute grade.
    expected = {
        "projected_length_m": 12.071067811865476,
        "dem_polyline_length_m": 12.558261413458432,
        "ascent_m": 3.3531494140625,
        "descent_m": 0,
        "net_elevation_change_m": 3.3531494140625,
        "max_terrain_slope_deg": 19.402825626556908,
        "mean_terrain_slope_deg": 18.459037517979905,
        "max_abs_step_grade_deg": 18.799394807511753,
    }
    assert result == pytest.approx(expected, abs=1e-12)


def test_statistics_reverse_route(statistics_patch):
    heights_m, slopes_deg, transform_m = statistics_patch
    # Include both upward and downward steps through the actual observations.
    cells = [[1, 1], [1, 2], [2, 3], [2, 2]]
    forward = route_statistics(cells, heights_m, slopes_deg, transform_m)
    backward = route_statistics(cells[::-1], heights_m, slopes_deg, transform_m)
    assert forward["ascent_m"] > 0 and forward["descent_m"] > 0
    assert forward["ascent_m"] == backward["descent_m"]
    assert forward["descent_m"] == backward["ascent_m"]
    assert forward["net_elevation_change_m"] == -backward["net_elevation_change_m"]
    assert forward["ascent_m"] - forward["descent_m"] == forward["net_elevation_change_m"]
    for key in ("projected_length_m", "dem_polyline_length_m", "max_terrain_slope_deg",
                "mean_terrain_slope_deg", "max_abs_step_grade_deg"):
        assert forward[key] == pytest.approx(backward[key], abs=1e-12)
    assert forward["dem_polyline_length_m"] >= forward["projected_length_m"]


def test_statistics_single_cell(statistics_patch):
    heights_m, slopes_deg, transform_m = statistics_patch
    result = route_statistics([[1, 1]], heights_m, slopes_deg, transform_m)
    for key in ("projected_length_m", "dem_polyline_length_m", "ascent_m", "descent_m",
                "net_elevation_change_m"):
        assert result[key] == 0
    # Same independently derived center inclination as test_statistics_hand_checked.
    assert result["mean_terrain_slope_deg"] == pytest.approx(17.646201418369948, abs=1e-12)
    assert result["max_terrain_slope_deg"] == result["mean_terrain_slope_deg"]
    assert result["max_abs_step_grade_deg"] is None


@pytest.mark.parametrize("cells", [
    [], [1, 1], [[1.0, 1.0]], [[-1, 1]], [[5, 1]],
    [[1, 1], [1, 1]], [[1, 1], [3, 3]], [[0, 0]],
])
def test_statistics_invalid_route(statistics_patch, cells):
    heights_m, slopes_deg, transform_m = statistics_patch
    with pytest.raises(ValueError):
        route_statistics(cells, heights_m, slopes_deg, transform_m)


def test_statistics_invalid_grids(statistics_patch):
    heights_m, slopes_deg, transform_m = statistics_patch
    with pytest.raises(ValueError, match="matching 2D"):
        route_statistics([[1, 1]], heights_m[:-1], slopes_deg, transform_m)
    masked_m = np.ma.array(heights_m, mask=False)
    masked_m.mask[1, 1] = True
    with pytest.raises(ValueError, match="invalid elevation"):
        route_statistics([[1, 1]], masked_m, slopes_deg, transform_m)
    with pytest.raises(ValueError, match="between 0 and 90"):
        route_statistics([[1, 1]], heights_m, -slopes_deg, transform_m)


@pytest.mark.parametrize("transform_m", [
    Affine(5, 1, 0, 0, -5, 0), Affine(0, 0, 0, 0, -5, 0),
    Affine(5, 0, np.nan, 0, -5, 0),
])
def test_statistics_invalid_transform(statistics_patch, transform_m):
    heights_m, slopes_deg, _ = statistics_patch
    with pytest.raises(ValueError):
        route_statistics([[1, 1]], heights_m, slopes_deg, transform_m)
