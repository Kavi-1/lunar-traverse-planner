"""Least-cost routes on an axis-aligned projected-meter raster, without an API."""

import numpy as np
import numpy.typing as npt
from rasterio import Affine
from rasterio.transform import rowcol, xy
from skimage.graph import MCP_Geometric


def route_statistics(
    route_rc: npt.ArrayLike,
    elevation_m: npt.ArrayLike,
    slope_deg: npt.ArrayLike,
    transform_m: Affine,
) -> dict[str, float | None]:
    """Measure an ordered eight-neighbor route on matching elevation/slope grids.

    Lengths, elevation changes, ascent, and descent are meters; slopes/grades
    are degrees. Terrain slope is the raster inclination, whereas step grade
    follows travel direction. A one-cell route has zero distance and elevation
    change, its cell slope as mean/max, and no step grade (None).

    DEM polyline length uses projected horizontal spacing and sampled height
    changes. It is an estimate, omitting reference-radius elevation scaling,
    curvature corrections, and unresolved terrain; see DECISIONS.md.
    """
    cells = np.asarray(route_rc)
    if cells.ndim != 2 or cells.shape[1] != 2 or len(cells) == 0:
        raise ValueError("route_rc must be a nonempty array of row/column pairs")
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("route_rc indices must be integers")
    heights_m = np.ma.asarray(elevation_m, dtype=np.float64).filled(np.nan)
    slopes_deg = np.ma.asarray(slope_deg, dtype=np.float64).filled(np.nan)
    if heights_m.ndim != 2 or heights_m.shape != slopes_deg.shape:
        raise ValueError("Elevation and slope must be matching 2D grids")
    if not np.isfinite(tuple(transform_m)).all():
        raise ValueError("Transform must be finite")
    if transform_m.b != 0 or transform_m.d != 0:
        raise ValueError("Rotated or sheared grids are not supported")
    if transform_m.a == 0 or transform_m.e == 0:
        raise ValueError("Pixel spacing must be nonzero")
    if np.any(cells < 0) or np.any(cells >= heights_m.shape):
        raise ValueError("route_rc contains indices outside the raster")
    # Convert after the bounds check so unsigned index differences cannot wrap.
    cells = cells.astype(np.int64)
    step_rc = np.diff(cells, axis=0)
    if np.any(np.max(np.abs(step_rc), axis=1) != 1):
        raise ValueError("Route steps must connect distinct eight-neighbor cells")
    rows, columns = cells.T
    route_heights_m = heights_m[rows, columns]
    route_slopes_deg = slopes_deg[rows, columns]
    if not np.isfinite(route_heights_m).all() or not np.isfinite(route_slopes_deg).all():
        raise ValueError("Route visits invalid elevation or slope")
    if np.any((route_slopes_deg < 0) | (route_slopes_deg > 90)):
        raise ValueError("Route slope must be between 0 and 90 degrees")

    # Columns follow affine X spacing; rows follow affine Y spacing (negative
    # at Site04). hypot measures unsigned projected horizontal step lengths.
    distance_m = np.hypot(step_rc[:, 1] * transform_m.a, step_rc[:, 0] * transform_m.e)
    height_change_m = np.diff(route_heights_m)
    projected_length_m = float(distance_m.sum())
    if len(cells) == 1:
        mean_slope_deg = float(route_slopes_deg[0])
        max_abs_grade_deg = None
    else:
        # Each edge spends half its distance in each endpoint cell. Use the
        # same distance weighting as MCP, without applying preference weights.
        edge_slope_deg = (route_slopes_deg[:-1] + route_slopes_deg[1:]) / 2
        mean_slope_deg = float(np.sum(distance_m * edge_slope_deg) / projected_length_m)
        # Positive grade is ascent in route order; atan2 takes rise/run in m/m.
        step_grade_deg = np.rad2deg(np.arctan2(height_change_m, distance_m))
        max_abs_grade_deg = float(np.max(np.abs(step_grade_deg)))
    return {
        "projected_length_m": projected_length_m,
        "dem_polyline_length_m": float(np.hypot(distance_m, height_change_m).sum()),
        "ascent_m": float(np.maximum(height_change_m, 0).sum()),
        "descent_m": float(np.maximum(-height_change_m, 0).sum()),
        "net_elevation_change_m": float(route_heights_m[-1] - route_heights_m[0]),
        "max_terrain_slope_deg": float(route_slopes_deg.max()),
        "mean_terrain_slope_deg": mean_slope_deg,
        "max_abs_step_grade_deg": max_abs_grade_deg,
    }


def find_route(
    cost_weights: npt.ArrayLike,
    transform_m: Affine,
    start_xy_m: tuple[float, float],
    goal_xy_m: tuple[float, float],
) -> dict[str, object]:
    """Return a pixel-center route and cost in weighted projected meters.

    cost_weights are dimensionless positive weights, with masked/nonfinite cells
    blocked. Coordinates and affine spacing are projected meters. Endpoints snap
    to their containing cell centers; blocked/outside endpoints raise ValueError.
    A disconnected goal returns status='no_route', with route and cost set to None.

    Eight-neighbor moves permit diagonal corner contact even if one or both side
    cells are blocked. Counts report those contacts, not physical clearance.
    MCP averages endpoint weights and multiplies by the step distance:
    https://scikit-image.org/docs/stable/api/skimage.graph.html#skimage.graph.MCP_Geometric
    Distances remain projected, as approved in DECISIONS.md.
    """
    weights = np.ma.asarray(cost_weights, dtype=np.float64).filled(np.nan)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("cost_weights must be a nonempty 2D array")
    finite = np.isfinite(weights)
    if np.any(finite & (weights <= 0)):
        raise ValueError("Finite cost_weights must be positive")
    weights = np.where(finite, weights, np.inf)
    if not np.isfinite(tuple(transform_m)).all():
        raise ValueError("Transform must be finite")
    if transform_m.b != 0 or transform_m.d != 0:
        raise ValueError("Rotated or sheared grids are not supported")
    if transform_m.a == 0 or transform_m.e == 0:
        raise ValueError("Pixel spacing must be nonzero")

    cells = []
    snapped_xy_m = []
    snap_distances_m = []
    for name, requested_xy_m in (("start", start_xy_m), ("goal", goal_xy_m)):
        if np.shape(requested_xy_m) != (2,) or not np.isfinite(requested_xy_m).all():
            raise ValueError(f"{name} must contain two finite projected coordinates in meters")
        # rasterio's inverse affine floors to the containing pixel. Its xy()
        # default adds the half-pixel offset, returning the DEM sample center.
        row, column = rowcol(transform_m, *requested_xy_m)
        if not (0 <= row < weights.shape[0] and 0 <= column < weights.shape[1]):
            raise ValueError(f"{name} is outside the raster")
        if not finite[row, column]:
            raise ValueError(f"{name} is blocked")
        center_xy_m = tuple(float(value) for value in xy(transform_m, row, column))
        cells.append((int(row), int(column)))
        snapped_xy_m.append(center_xy_m)
        snap_distances_m.append(float(np.hypot(
            center_xy_m[0] - requested_xy_m[0], center_xy_m[1] - requested_xy_m[1]
        )))

    result = {
        "status": "no_route",
        "requested_start_xy_m": tuple(start_xy_m),
        "requested_goal_xy_m": tuple(goal_xy_m),
        "snapped_start_xy_m": snapped_xy_m[0],
        "snapped_goal_xy_m": snapped_xy_m[1],
        "start_snap_distance_m": snap_distances_m[0],
        "goal_snap_distance_m": snap_distances_m[1],
        "route_rc": None,
        "route_xy_m": None,
        "total_cost_weighted_m": None,
        "diagonal_blocked_side_steps": None,
        "diagonal_both_blocked_steps": None,
    }
    if cells[0] == cells[1]:
        route_rc = np.asarray([cells[0]], dtype=np.int64)
        total_cost_weighted_m = 0.0
    else:
        # Sampling order is row, column. Signed Y spacing locates coordinates;
        # positive magnitudes measure distance, including sqrt(dx²+dy²) diagonals.
        solver = MCP_Geometric(
            weights, fully_connected=True,
            sampling=(abs(transform_m.e), abs(transform_m.a)),
        )
        cumulative_costs, _ = solver.find_costs([cells[0]], [cells[1]])
        total_cost_weighted_m = float(cumulative_costs[cells[1]])
        if not np.isfinite(total_cost_weighted_m):
            return result
        route_rc = np.asarray(solver.traceback(cells[1]), dtype=np.int64)

    rows, columns = route_rc.T
    x_m, y_m = xy(transform_m, rows, columns)
    route_xy_m = np.column_stack((x_m, y_m))
    step_rc = np.diff(route_rc, axis=0)
    step_distances_m = np.hypot(step_rc[:, 0] * transform_m.e, step_rc[:, 1] * transform_m.a)
    route_weights = weights[rows, columns]
    summed_cost_weighted_m = np.sum(
        step_distances_m * (route_weights[:-1] / 2 + route_weights[1:] / 2)
    )
    if not np.isclose(summed_cost_weighted_m, total_cost_weighted_m, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Traceback edge costs disagree with MCP cumulative cost")

    diagonal = np.all(step_rc != 0, axis=1)
    before = route_rc[:-1][diagonal]
    after = route_rc[1:][diagonal]
    # The two side cells share the diagonal's corner. Neither is visited by
    # the center-to-center step, so ordinary MCP does not check their costs.
    first_blocked = ~finite[before[:, 0], after[:, 1]]
    second_blocked = ~finite[after[:, 0], before[:, 1]]
    result.update(
        status="ok",
        route_rc=route_rc,
        route_xy_m=route_xy_m,
        total_cost_weighted_m=total_cost_weighted_m,
        diagonal_blocked_side_steps=int(np.count_nonzero(first_blocked | second_blocked)),
        diagonal_both_blocked_steps=int(np.count_nonzero(first_blocked & second_blocked)),
    )
    return result
