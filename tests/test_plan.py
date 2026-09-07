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


def test_clone_geometry_hand_checked():
    from core.clones import geometry_distances_m, vertex_weights_m

    # Coordinate-only geometry, no fabricated terrain. Edges 3 and 4 m give
    # weights 1.5, 3.5, 2 m. Parallel two-point routes at 4 m separation.
    np.testing.assert_array_equal(vertex_weights_m([[0, 0], [3, 0], [3, 4]]), [1.5, 3.5, 2])
    first_m = [[0, 0], [3, 0]]
    second_m = [[0, 4], [3, 4]]
    assert geometry_distances_m(first_m, first_m) == {
        'mean_nearest_distance_m': 0, 'max_nearest_distance_m': 0}
    expected = {'mean_nearest_distance_m': 4, 'max_nearest_distance_m': 4}
    assert geometry_distances_m(first_m, second_m) == expected
    assert geometry_distances_m(second_m, first_m) == expected
    # Unequal collinear lengths 2 and 4: directed weighted means 0 and 0.5,
    # symmetric mean 0.25, maximum 2. Middle vertices carry two half edges.
    short_m = [[0, 0], [2, 0]]
    long_m = [[0, 0], [2, 0], [4, 0]]
    expected = {'mean_nearest_distance_m': 0.25, 'max_nearest_distance_m': 2}
    assert geometry_distances_m(short_m, long_m) == expected
    assert geometry_distances_m(long_m, short_m) == expected
    assert geometry_distances_m([[0, 0]], [[3, 4]]) == {
        'mean_nearest_distance_m': 5, 'max_nearest_distance_m': 5}
    with pytest.raises(ValueError):
        vertex_weights_m([])


def test_clone_summaries_hand_checked():
    from core.clones import continuous_summary

    assert continuous_summary([1, 2, 3]) == pytest.approx({
        'count': 3, 'mean': 2, 'sample_std': 1, 'median': 2,
        'p5': 1.1, 'p95': 2.9, 'min': 1, 'max': 3})
    assert continuous_summary([])['count'] == 0
    assert continuous_summary([])['mean'] is None
    assert continuous_summary([7])['sample_std'] is None


def test_clone_cutoff_and_missing_terrain(statistics_patch):
    from core.clones import nominal_feasibility

    elevation_m, slope_deg, _ = statistics_patch
    cells = np.array([[1, 1], [1, 2]])
    cutoff_deg = float(slope_deg[1, 1])
    result = nominal_feasibility(cells, elevation_m, slope_deg, cutoff_deg)
    assert result['cutoff_violation_cells'] == 1
    assert result['max_exceedance_deg'] == pytest.approx(18.242866980426832-cutoff_deg)
    assert result['missing_terrain_cells'] == 0
    invalid_deg = slope_deg.copy()
    invalid_deg[1, 2] = np.nan
    result = nominal_feasibility(cells, elevation_m, invalid_deg, cutoff_deg)
    assert result['cutoff_violation_cells'] == 0
    assert result['missing_terrain_cells'] == 1
    assert not result['terrain_feasible']


@pytest.fixture(scope='module')
def clone_case():
    import rasterio

    from core.clones import solve_clone, terrain_window_m

    elevation_m, slope_deg, transform_m = terrain_window_m(DEM_PATH)
    with rasterio.open(DEM_PATH.parent / 'site04_mean_local_shadow.tif') as source:
        shadow_fraction = source.read(1)
    nominal = solve_clone(elevation_m, slope_deg, shadow_fraction, transform_m, None)
    return elevation_m, slope_deg, shadow_fraction, transform_m, np.array(nominal['route_rc'])


def test_clone_nominal_reproduction_and_fixed_cost(clone_case):
    from core.clones import solve_clone

    record = solve_clone(*clone_case)
    assert record['statistics']['projected_length_m'] == pytest.approx(1858.4419177103414)
    assert record['statistics']['total_cost_weighted_m'] == pytest.approx(1947.5175345932507)
    assert record['statistics']['distance_weighted_mean_local_shadow_fraction'] == pytest.approx(
        0.002958036734429219)
    assert record['statistics']['diagonal_blocked_side_steps'] == 1
    assert record['exact_nominal_match']
    assert record['cost_disadvantage_weighted_m'] == 0
    assert record['cost_disadvantage_percent'] == 0


def test_clone_failures_and_aggregate_denominators(clone_case):
    from core.clones import aggregate_records, solve_clone

    elevation_m, slope_deg, shadow_fraction, transform_m, nominal_rc = clone_case
    success = solve_clone(*clone_case)
    blocked_deg = slope_deg.copy()
    blocked_deg[200, 80] = np.nan  # Mask the actual start observation.
    blocked = solve_clone(elevation_m, blocked_deg, shadow_fraction, transform_m, nominal_rc)
    assert blocked['status'] == 'blocked_endpoints'
    assert blocked['blocked_endpoints'] == ['start']
    disconnected_deg = slope_deg.copy()
    disconnected_deg[:, 180] = np.nan  # Mask a complete separating column.
    disconnected = solve_clone(elevation_m, disconnected_deg, shadow_fraction,
                               transform_m, nominal_rc)
    assert disconnected['status'] == 'no_route'
    for record in (blocked, disconnected):
        for key in ('route_rc', 'statistics', 'geometry', 'nominal_statistics',
                    'cost_disadvantage_weighted_m', 'cost_disadvantage_percent'):
            assert record[key] is None
    aggregate = aggregate_records([success, blocked, disconnected])
    assert aggregate['ensemble_count'] == 3
    assert aggregate['successful_routes'] == 1
    assert aggregate['blocked_endpoints'] == 1
    assert aggregate['disconnected_goals'] == 1
    assert aggregate['statistics']['projected_length_m']['count'] == 1
    assert aggregate['clones_with_missing_terrain_cells'] == 2


def test_fixed_route_measurements_hand_checked(statistics_patch):
    import rasterio

    from core.clones import route_measurements

    heights_m, slopes_deg, transform_m = statistics_patch
    cells = np.array([[1, 1], [1, 2], [2, 3]])
    with rasterio.open(DEM_PATH.parent / 'site04_mean_local_shadow.tif') as source:
        shadow_fraction = source.read(1)[:5, :5]
    costs = build_cost_surface(slopes_deg, slope_limit_deg=25, slope_weight=2,
                               shadow_fraction=shadow_fraction, shadow_weight=2)
    result = route_measurements(cells, heights_m, slopes_deg, shadow_fraction, costs, transform_m)
    # Independent scalar edge sums, with distances 5 and sqrt(50) meters.
    weights = [1 + 2*math.tan(math.radians(deg))**2 + 2*shadow_fraction[r, c]
               for (r, c), deg in zip(cells, [17.646201418369948,
                                             18.242866980426832, 19.402825626556908])]
    expected_cost_m = 5*(weights[0]+weights[1])/2 + math.sqrt(50)*(weights[1]+weights[2])/2
    expected_exposure = (5*(shadow_fraction[1, 1]+shadow_fraction[1, 2])/2
                         + math.sqrt(50)*(shadow_fraction[1, 2]+shadow_fraction[2, 3])/2)
    assert result['total_cost_weighted_m'] == pytest.approx(expected_cost_m, abs=1e-12)
    assert result['distance_weighted_mean_local_shadow_fraction'] == pytest.approx(
        expected_exposure/(5+math.sqrt(50)), abs=1e-12)


def test_feasible_reference_cost_disadvantage(clone_case):
    from itertools import pairwise

    from core.clones import GOAL_XY_M, START_XY_M, solve_clone

    elevation_m, slope_deg, shadow_fraction, transform_m, _ = clone_case
    # Use a different feasible route through the real DEM to exercise nonzero
    # disadvantage: the slope-only optimum omits the existing shadow preference.
    slope_costs = build_cost_surface(slope_deg, slope_limit_deg=20, slope_weight=2)
    reference = find_route(slope_costs, transform_m, START_XY_M, GOAL_XY_M)['route_rc']
    record = solve_clone(elevation_m, slope_deg, shadow_fraction, transform_m, reference)
    assert record['nominal_feasibility']['terrain_feasible']
    # Independent scalar trapezoidal cost from the actual sampled slopes/shadow.
    reference_cost_m = 0.0
    for (r0, c0), (r1, c1) in pairwise(reference):
        distance_m = math.hypot((r1-r0)*5, (c1-c0)*5)
        before = 1 + 2*math.tan(math.radians(slope_deg[r0, c0]))**2 + 2*shadow_fraction[r0, c0]
        after = 1 + 2*math.tan(math.radians(slope_deg[r1, c1]))**2 + 2*shadow_fraction[r1, c1]
        reference_cost_m += distance_m*(before+after)/2
    optimum_m = 1947.5175345932507
    assert record['cost_disadvantage_weighted_m'] == pytest.approx(reference_cost_m-optimum_m)
    assert record['cost_disadvantage_percent'] == pytest.approx(
        100*(reference_cost_m-optimum_m)/optimum_m)
    assert record['cost_disadvantage_weighted_m'] > 0


def test_occupancy_fraction_hand_checked():
    from core.clones import route_occupancy_fraction

    # Route cells only, no terrain. Shared cell is visited by both routes;
    # the repeated visit in the first route must count only once.
    first_rc = np.array([[0, 0], [0, 1], [0, 0]])
    second_rc = np.array([[0, 0], [1, 0]])
    np.testing.assert_array_equal(route_occupancy_fraction([first_rc, second_rc], (2, 2)),
                                 [[1, 0.5], [0.5, 0]])
    assert np.isnan(route_occupancy_fraction([], (2, 2))).all()


def test_nominal_cutoff_severity_and_fraction(clone_case):
    from core.clones import aggregate_records, solve_clone, terrain_window_m

    _, _, shadow_fraction, _, nominal_rc = clone_case
    clone_path = DEM_PATH.parent / 'Clones/Site04_final_adj_5mpp_0001_err.tif'
    elevation_m, slope_deg, transform_m = terrain_window_m(clone_path)
    clone = solve_clone(elevation_m, slope_deg, shadow_fraction, transform_m, nominal_rc)
    nominal = solve_clone(*clone_case)
    # Actual clone 0001 has one offending cell; its slope is 21.42163299392544°.
    case = clone['nominal_cutoff_sensitivity']['20']
    assert case['cutoff_violation_cells'] == 1
    assert case['cutoff_violation_fraction'] == pytest.approx(1 / len(nominal_rc))
    assert case['max_exceedance_deg'] == pytest.approx(1.4216329939254386)
    for cutoff in ('22', '25'):
        assert clone['nominal_cutoff_sensitivity'][cutoff]['cutoff_violation_cells'] == 0
        assert clone['nominal_cutoff_sensitivity'][cutoff]['terrain_feasible']
    summary = aggregate_records([nominal, clone])['nominal_cutoff_sensitivity']
    assert summary['20']['ensemble_count'] == 2
    assert summary['20']['clones_with_cutoff_violations'] == 1
    severity = summary['20']['violating_clones_only']['max_exceedance_deg']
    assert severity['count'] == 1  # Do not dilute severity with the feasible case's zero.
    assert severity['mean'] == pytest.approx(1.4216329939254386)
    assert summary['20']['all_clones']['cutoff_violation_fraction']['mean'] == pytest.approx(
        0.5 / len(nominal_rc))
    assert summary['25']['violating_clones_only']['max_exceedance_deg']['count'] == 0
    assert summary['25']['violating_clones_only']['max_exceedance_deg']['mean'] is None


def test_fifteen_degree_nominal_plan(clone_case):
    from core.clones import GOAL_XY_M, START_XY_M, nominal_feasibility

    elevation_m, slope_deg, shadow_fraction, transform_m, _ = clone_case
    costs = build_cost_surface(slope_deg, slope_limit_deg=15, slope_weight=2,
                               shadow_fraction=shadow_fraction, shadow_weight=2)
    result = find_route(costs, transform_m, START_XY_M, GOAL_XY_M)
    assert result['status'] == 'ok'
    cells = result['route_rc']
    stats = route_statistics(cells, elevation_m, slope_deg, transform_m)
    # The downloaded nominal route has 164 axial 5 m steps and 151 diagonal
    # sqrt(50) m steps: 164*5 + 151*sqrt(50) = 1887.7312395916867 m.
    steps_rc = np.diff(cells, axis=0)
    diagonal_count = np.count_nonzero(np.all(steps_rc != 0, axis=1))
    axial_count = len(steps_rc) - diagonal_count
    assert axial_count == 164
    assert diagonal_count == 151
    expected_length_m = 164*5 + 151*math.sqrt(50)
    assert stats['projected_length_m'] == pytest.approx(expected_length_m)
    assert stats['projected_length_m'] == pytest.approx(1887.7312395916867)
    assert nominal_feasibility(cells, elevation_m, slope_deg, 15)['terrain_feasible']
    assert stats['max_terrain_slope_deg'] == pytest.approx(13.453593067870676)
    from core.clones import terrain_window_m

    clone_m, clone_deg, _ = terrain_window_m(
        DEM_PATH.parent / 'Clones/Site04_final_adj_5mpp_0023_err.tif')
    feasibility = nominal_feasibility(cells, clone_m, clone_deg, 20)
    # Downloaded clone cell (123,252) in the analysis crop: opposite-neighbor
    # differences divided by signed 10 m baselines give central gradients.
    gradient_x = (1414.2200927734375 - 1410.384033203125) / 10
    gradient_y = (1411.85595703125 - 1412.0333251953125) / -10
    expected_slope_deg = math.degrees(math.atan(math.hypot(gradient_x, gradient_y)))
    assert expected_slope_deg == pytest.approx(21.00757773389859)
    assert clone_deg[123, 252] == pytest.approx(expected_slope_deg)
    assert feasibility['cutoff_violation_cells'] == 1
    assert feasibility['max_exceedance_deg'] == pytest.approx(expected_slope_deg - 20)
    assert feasibility['cutoff_violation_fraction'] == pytest.approx(1 / 316)
    assert feasibility['missing_terrain_cells'] == 0
