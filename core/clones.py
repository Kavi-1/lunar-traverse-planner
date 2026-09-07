"""Site04 clone sensitivity conditional on saved nominal local illumination.

The fixed bounds, endpoints, 20-degree cutoff and weights are analysis choices,
not physical defaults. PGDA clones are elevations, NOT errors to add to the DEM.
See PROJECT_SPEC.md and DECISIONS.md for source context and model limitations.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.windows import from_bounds

from core.cost import build_cost_surface
from core.plan import find_route, route_statistics
from core.terrain import compute_slope_deg, load_dem_m

START_XY_M = (-5697.5, -10002.5)
GOAL_XY_M = (-4497.5, -10002.5)
BOUNDS_M = (-6100, -11000, -4100, -9000)
START_UTC = "2026-09-01T09:00:00"
END_UTC = "2026-09-01T15:00:00"


def matching_projection(source_crs: CRS | None, nominal_crs: CRS) -> bool:
    """Compare Site04 horizontal projection parameters, ignoring descriptive names.

    The nominal WKT calls its sphere 'unnamed', while clones call it 'unknown'.
    GDAL CRS equality rejects those labels even though the radius, pole,
    meridian, offsets and units agree. Compare the PROJ parameter dictionaries;
    retain original WKT in the report. This checks geometry, not frame provenance.
    """
    return (source_crs is not None and source_crs.is_projected
            and source_crs.to_dict() == nominal_crs.to_dict())


def input_hash(path: Path, *, recorded: bool = True) -> str:
    """Hash bytes; require the downloader's SHA-256 sidecar for source DEMs."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if recorded and path.with_suffix(path.suffix + ".sha256").read_text().split()[0] != digest:
        raise ValueError(f"Hash mismatch: {path}")
    return digest


def terrain_window_m(path: Path) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Load elevations in meters, slope in degrees, and transform in meters.

    A one-pixel halo preserves all nine samples required by the slope validity
    convention. Copy the crop so the full DEM is not retained between clones.
    """
    elevation_m, transform_m, _ = load_dem_m(path)
    window = from_bounds(*BOUNDS_M, transform_m)
    row, col = int(window.row_off), int(window.col_off)
    if row < 1 or col < 1 or row + 401 > len(elevation_m) or col + 401 > elevation_m.shape[1]:
        raise ValueError("Analysis window needs a complete one-pixel halo")
    halo_m = elevation_m[row-1:row+401, col-1:col+401]
    slope_deg = compute_slope_deg(halo_m, transform_m)[1:-1, 1:-1].copy()
    return halo_m[1:-1, 1:-1].copy(), slope_deg, rasterio.windows.transform(window, transform_m)


def vertex_weights_m(route_xy_m: npt.ArrayLike) -> np.ndarray:
    """Half of each adjacent projected edge length at every vertex, in meters."""
    points_m = np.asarray(route_xy_m, dtype=float)
    if points_m.ndim != 2 or points_m.shape[1] != 2 or len(points_m) == 0:
        raise ValueError("Expected nonempty Nx2 projected coordinates in meters")
    if not np.isfinite(points_m).all():
        raise ValueError("Coordinates must be finite")
    edge_m = np.linalg.norm(np.diff(points_m, axis=0), axis=1)
    weights_m = np.zeros(len(points_m))
    weights_m[:-1] += edge_m / 2
    weights_m[1:] += edge_m / 2
    return weights_m


def geometry_distances_m(first_xy_m: npt.ArrayLike, second_xy_m: npt.ArrayLike) -> dict:
    """Symmetric nearest sampled pixel-center distances in meters, not polylines.

    Average the two directed distance-weighted means equally. Hausdorff is the
    larger directed maximum. A zero-length route uses its unweighted point mean.
    """
    first_weights_m = vertex_weights_m(first_xy_m)
    second_weights_m = vertex_weights_m(second_xy_m)
    distances_m = np.linalg.norm(
        np.asarray(first_xy_m)[:, None, :] - np.asarray(second_xy_m)[None, :, :], axis=2)
    first_m, second_m = distances_m.min(axis=1), distances_m.min(axis=0)
    means_m = []
    for values_m, weights_m in ((first_m, first_weights_m), (second_m, second_weights_m)):
        if weights_m.sum() > 0:
            means_m.append(float(np.average(values_m, weights=weights_m)))
        else:
            means_m.append(float(values_m.mean()))
    return {"mean_nearest_distance_m": sum(means_m) / 2,
            "max_nearest_distance_m": float(max(first_m.max(), second_m.max()))}


def continuous_summary(values: list[float]) -> dict:
    """Descriptive summary in input units; sample SD and linear percentiles."""
    if not values:
        return dict.fromkeys(("mean", "sample_std", "median", "p5", "p95", "min", "max")) | {
            "count": 0}
    data = np.asarray(values, dtype=float)
    return {"count": len(data), "mean": float(data.mean()),
            "sample_std": float(data.std(ddof=1)) if len(data) > 1 else None,
            "median": float(np.median(data)), "p5": float(np.percentile(data, 5, method="linear")),
            "p95": float(np.percentile(data, 95, method="linear")),
            "min": float(data.min()), "max": float(data.max())}


def nominal_feasibility(route_rc: np.ndarray, elevation_m: np.ndarray,
                        slope_deg: np.ndarray, slope_limit_deg: float) -> dict:
    """Count missing terrain separately from strict cutoff exceedances in degrees."""
    rows, cols = route_rc.T
    values_deg = slope_deg[rows, cols]
    missing = ~np.isfinite(values_deg) | ~np.isfinite(elevation_m[rows, cols])
    exceeds = ~missing & (values_deg > slope_limit_deg)
    return {"missing_terrain_cells": int(missing.sum()),
            "route_cell_count": len(route_rc),
            "cutoff_violation_cells": int(exceeds.sum()),
            # Count samples, not distance: every ordered nominal-route cell,
            # including endpoints and missing samples, remains in the denominator.
            "cutoff_violation_fraction": float(exceeds.sum() / len(route_rc)),
            "max_exceedance_deg": float(np.max(values_deg[exceeds] - slope_limit_deg))
            if exceeds.any() else 0.0,
            "terrain_feasible": bool(not missing.any() and not exceeds.any())}


def route_measurements(route_rc: np.ndarray, elevation_m: np.ndarray, slope_deg: np.ndarray,
                       shadow_fraction: np.ndarray, cost_weights: np.ndarray,
                       transform_m: Affine) -> dict:
    """Evaluate a fixed route: meters, degrees, weighted meters and shadow fraction."""
    result = route_statistics(route_rc, elevation_m, slope_deg, transform_m)
    # Pixel centers: X follows columns, Y follows signed (negative) row spacing.
    x_m, y_m = rasterio.transform.xy(transform_m, *route_rc.T)
    weights_m = vertex_weights_m(np.column_stack((x_m, y_m)))
    rows, cols = route_rc.T
    result["total_cost_weighted_m"] = float(np.dot(weights_m, cost_weights[rows, cols]))
    result["distance_weighted_mean_local_shadow_fraction"] = float(
        np.average(shadow_fraction[rows, cols], weights=weights_m)
        if weights_m.sum() else shadow_fraction[rows[0], cols[0]])
    step_rc = np.diff(route_rc, axis=0)
    diagonal = np.all(step_rc != 0, axis=1)
    before, after = route_rc[:-1][diagonal], route_rc[1:][diagonal]
    first = ~np.isfinite(cost_weights[before[:, 0], after[:, 1]])
    second = ~np.isfinite(cost_weights[after[:, 0], before[:, 1]])
    result["diagonal_blocked_side_steps"] = int((first | second).sum())
    result["diagonal_both_blocked_steps"] = int((first & second).sum())
    return result


def solve_clone(elevation_m: np.ndarray, slope_deg: np.ndarray, shadow_fraction: np.ndarray,
                transform_m: Affine, nominal_rc: np.ndarray | None) -> dict:
    """Optimize the fixed Site04 case and evaluate its nominal route on this clone."""
    timings = {}
    started_seconds = perf_counter()
    costs = build_cost_surface(slope_deg, slope_limit_deg=20, slope_weight=2,
                               shadow_fraction=shadow_fraction, shadow_weight=2)
    timings["cost_seconds"] = perf_counter() - started_seconds
    blocked = [name for name, point_m in (("start", START_XY_M), ("goal", GOAL_XY_M))
               if not np.isfinite(costs[rasterio.transform.rowcol(transform_m, *point_m)])]
    started_seconds = perf_counter()
    route = (find_route(costs, transform_m, START_XY_M, GOAL_XY_M) if not blocked else
             {"status": "blocked_endpoints", "route_rc": None})
    timings["routing_seconds"] = perf_counter() - started_seconds
    result = {"status": route["status"], "blocked_endpoints": blocked,
              "route_rc": None, "statistics": None, "geometry": None,
              "exact_nominal_match": None, "nominal_statistics": None,
              "cost_disadvantage_weighted_m": None, "cost_disadvantage_percent": None,
              "timings": timings}
    started_seconds = perf_counter()
    if route["status"] == "ok":
        cells = route["route_rc"]
        result["route_rc"] = cells.tolist()
        result["statistics"] = route_measurements(
            cells, elevation_m, slope_deg, shadow_fraction, costs, transform_m)
        if nominal_rc is not None:
            result["exact_nominal_match"] = bool(np.array_equal(cells, nominal_rc))
            nx_m, ny_m = rasterio.transform.xy(transform_m, *nominal_rc.T)
            result["geometry"] = geometry_distances_m(
                route["route_xy_m"], np.column_stack((nx_m, ny_m)))
    if nominal_rc is not None:
        # Evaluate the unchanged 20-degree plan at user-selected allowable
        # cutoffs; do not replan or claim a calibrated safety margin.
        result["nominal_cutoff_sensitivity"] = {
            str(cutoff_deg): nominal_feasibility(nominal_rc, elevation_m, slope_deg, cutoff_deg)
            for cutoff_deg in (20, 22, 25)}
        feasibility = result["nominal_cutoff_sensitivity"]["20"].copy()
        feasibility["missing_shadow_cells"] = int(
            (~np.isfinite(shadow_fraction[tuple(nominal_rc.T)])).sum())
        result["nominal_feasibility"] = feasibility
        if feasibility["terrain_feasible"] and not feasibility["missing_shadow_cells"]:
            result["nominal_statistics"] = route_measurements(
                nominal_rc, elevation_m, slope_deg, shadow_fraction, costs, transform_m)
            if result["statistics"] is None:
                raise RuntimeError("Feasible nominal route but optimizer failed")
            optimum_m = result["statistics"]["total_cost_weighted_m"]
            difference_m = result["nominal_statistics"]["total_cost_weighted_m"] - optimum_m
            if difference_m < -1e-8:
                raise RuntimeError("Nominal route cheaper than computed optimum")
            result["cost_disadvantage_weighted_m"] = max(0.0, difference_m)
            result["cost_disadvantage_percent"] = 100 * max(0.0, difference_m) / optimum_m
    timings["metrics_seconds"] = perf_counter() - started_seconds
    return result


def aggregate_records(records: list[dict]) -> dict:
    """Keep ensemble counts unconditional; route distributions require success."""
    successful = [record for record in records if record["status"] == "ok"]
    result = {"ensemble_count": len(records), "successful_routes": len(successful),
              "blocked_endpoints": sum(r["status"] == "blocked_endpoints" for r in records),
              "disconnected_goals": sum(r["status"] == "no_route" for r in records),
              "exact_nominal_matches": sum(r["exact_nominal_match"] is True for r in records)}
    result["clones_invalidating_nominal_terrain_route"] = sum(
        not r["nominal_feasibility"]["terrain_feasible"] for r in records)
    for key in ("cutoff_violation_cells", "missing_terrain_cells", "missing_shadow_cells"):
        result[f"clones_with_{key}"] = sum(r["nominal_feasibility"][key] > 0 for r in records)
    for group in ("statistics", "geometry", "nominal_statistics", "nominal_feasibility"):
        keys = {key for r in records if r.get(group) for key, value in r[group].items()
                if not isinstance(value, bool)}
        result[group] = {key: continuous_summary([r[group][key] for r in records
                        if r.get(group) and r[group].get(key) is not None]) for key in sorted(keys)}
    for key in ("cost_disadvantage_weighted_m", "cost_disadvantage_percent"):
        result[key] = continuous_summary([r[key] for r in records if r[key] is not None])
    result["nominal_cutoff_sensitivity"] = {}
    for cutoff_deg in (20, 22, 25):
        cases = [r["nominal_cutoff_sensitivity"][str(cutoff_deg)] for r in records]
        invalidating = [case for case in cases if case["cutoff_violation_cells"] > 0]
        result["nominal_cutoff_sensitivity"][str(cutoff_deg)] = {
            "ensemble_count": len(cases),
            "clones_with_cutoff_violations": len(invalidating),
            "terrain_feasible_clones": sum(case["terrain_feasible"] for case in cases),
            "clones_with_missing_terrain": sum(case["missing_terrain_cells"] > 0 for case in cases),
            "all_clones": {key: continuous_summary([case[key] for case in cases])
                           for key in ("cutoff_violation_cells", "cutoff_violation_fraction")},
            "violating_clones_only": {
                key: continuous_summary([case[key] for case in invalidating])
                for key in ("max_exceedance_deg", "cutoff_violation_cells",
                            "cutoff_violation_fraction")}}
    return result






def run_site04(data_dir: Path) -> dict:
    """Preflight all 100 inputs, then write JSON; timing units are seconds."""
    started_seconds = perf_counter()
    nominal_path = data_dir / "Site04_final_adj_5mpp_surf.tif"
    paths = [data_dir / "Clones" / f"Site04_final_adj_5mpp_{i:04d}_err.tif"
             for i in range(1, 101)]
    shadow_path = data_dir / "site04_mean_local_shadow.tif"
    illumination_path = data_dir / "site04_illumination_report.json"
    for path in [nominal_path, shadow_path, illumination_path, *paths]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; download inputs before analysis")
    hashes = {str(path): input_hash(path) for path in [nominal_path, *paths]}
    source_crs_wkt = {}
    with rasterio.open(nominal_path) as base:
        if (base.crs is None or not base.crs.is_projected
                or base.crs.linear_units_factor[1] != 1.0):
            raise ValueError("Nominal CRS must be projected in meters")
        crs_wkt = base.crs.to_wkt()
        if base.shape != (3200, 3200) or base.transform != Affine(5, 0, -9000, 0, -5, 1000):
            raise ValueError("Unexpected nominal Site04 grid")
        for path in [nominal_path, *paths]:
            with rasterio.open(path) as source:
                if (source.count != 1 or source.shape != base.shape
                        or not matching_projection(source.crs, base.crs)
                        or source.transform != base.transform or source.scales != (1.0,)
                        or source.offsets != (0.0,) or source.dtypes != ("float32",)
                        or source.units[0] not in (None, "m", "meter", "metre", "meters", "metres")
                        or source.nodata is None or not np.isnan(source.nodata)):
                    raise ValueError(f"Unexpected raster registration or scale/offset: {path}")
                source_crs_wkt[str(path)] = source.crs.to_wkt()
        expected_transform_m = rasterio.windows.transform(
            from_bounds(*BOUNDS_M, base.transform), base.transform)
        with rasterio.open(shadow_path) as source:
            tags = source.tags()
            if (source.count != 1 or source.shape != (400, 400) or source.crs != base.crs
                    or source.transform != expected_transform_m or source.scales != (1.0,)
                    or source.offsets != (0.0,) or tags.get("start_utc") != START_UTC
                    or tags.get("end_utc") != END_UTC or tags.get("frame") != "MOON_ME_DE421"):
                raise ValueError("Saved illumination metadata does not match experiment")
            shadow_fraction = source.read(1, masked=True).filled(np.nan)
    if not np.isfinite(shadow_fraction).all() or np.any(
            (shadow_fraction < 0) | (shadow_fraction > 1)):
        raise ValueError("Expected complete saved shadow fractions in [0,1]")
    illumination = json.loads(illumination_path.read_text())
    expected = {"start_utc": START_UTC, "end_utc": END_UTC, "output_shape": [400, 400],
                "output_bounds_m": list(BOUNDS_M), "dem_sha256": hashes[str(nominal_path)],
                "slope_weight": 2, "shadow_weight": 2, "frame": "MOON_ME_DE421"}
    if any(illumination.get(key) != value for key, value in expected.items()):
        raise ValueError("Illumination report does not match nominal inputs")
    for key, expected_value in (("time_step_seconds", illumination["time_step_seconds"]),
                                ("distance_step_m", illumination["nominal"]["distance_step_m"]),
                                ("radius_m", illumination["nominal"]["radius_m"])):
        if float(tags.get(key, "nan")) != expected_value:
            raise ValueError(f"Illumination sampling metadata mismatch: {key}")
    if (illumination["unknown_pixels"] != 0 or not np.isclose(
            shadow_fraction.mean(), illumination["mean_shadow_fraction"], rtol=0, atol=1e-12)):
        raise ValueError("Illumination pixels disagree with saved summary")
    hashes.update({str(p): input_hash(p, recorded=False) for p in [shadow_path, illumination_path]})
    preflight_seconds = perf_counter() - started_seconds
    nominal_m, nominal_deg, transform_m = terrain_window_m(nominal_path)
    nominal = solve_clone(nominal_m, nominal_deg, shadow_fraction, transform_m, None)
    if nominal["status"] != "ok":
        raise RuntimeError("Nominal route failed")
    saved = next(r for r in illumination["routes"] if r["shadow_weight"] == 2)
    for key, expected_value in (("projected_length_m", 1858.441918),
                                ("total_cost_weighted_m", 1947.517535)):
        if not np.isclose(nominal["statistics"][key], expected_value, rtol=0, atol=1e-6):
            raise RuntimeError(f"Nominal {key} no longer reproduces milestone 3")
        if not np.isclose(nominal["statistics"][key], saved[key], rtol=0, atol=1e-8):
            raise RuntimeError(f"Nominal {key} disagrees with saved summary")
    nominal_rc = np.asarray(nominal["route_rc"])
    records = []
    for index, path in enumerate(paths, 1):
        load_started_seconds = perf_counter()
        elevation_m, slope_deg, _ = terrain_window_m(path)
        load_seconds = perf_counter() - load_started_seconds
        record = solve_clone(elevation_m, slope_deg, shadow_fraction, transform_m, nominal_rc)
        record.update(clone_id=f"{index:04d}", input_path=str(path), sha256=hashes[str(path)])
        record["timings"]["load_and_slope_seconds"] = load_seconds
        error_m = elevation_m - nominal_m
        record["elevation_difference_m"] = continuous_summary(error_m[np.isfinite(error_m)].tolist())
        records.append(record)
        print(f"Clone {index:04d}/0100: {record['status']}", flush=True)
    report = {"settings": {"site": "04", "start_xy_m": START_XY_M, "goal_xy_m": GOAL_XY_M,
              "bounds_m": BOUNDS_M, "shape": [400, 400], "transform_m": list(transform_m),
              "crs_wkt": crs_wkt, "clone_values": "full elevations in meters",
              "route_rc_convention": "ordered row,column cells in the cropped analysis grid",
              "slope_limit_deg": 20, "slope_weight": 2, "shadow_weight": 2,
              "nominal_evaluation_cutoffs_deg": [20, 22, 25],
              "start_utc": START_UTC, "end_utc": END_UTC, "illumination": "fixed nominal",
              "geometry": "sampled pixel centers; symmetric edge-weighted nearest distance",
              "percentiles": "numpy linear", "standard_deviation": "sample, ddof=1"},
              "input_sha256": hashes, "source_crs_wkt": source_crs_wkt,
              "software_versions": {"python": platform.python_version(),
              **{name: importlib.metadata.version(name) for name in
                 ("numpy", "rasterio", "scikit-image", "matplotlib")}},
              "nominal": nominal, "clones": records, "aggregate": aggregate_records(records),
              "timings": {"preflight_seconds": preflight_seconds}}
    report["timings"]["clone_stage_totals_seconds"] = {
        key: sum(record["timings"][key] for record in records) for key in records[0]["timings"]}
    report["timings"]["total_seconds_excluding_download"] = perf_counter() - started_seconds
    output_path = data_dir / "site04_clones_report.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=["04"], required=True)
    args = parser.parse_args()
    run_site04(Path("data/Site04"))
