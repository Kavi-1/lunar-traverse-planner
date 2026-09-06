"""Run the 15/20/25-degree sensitivity sweep on the full Site04 grid.

From the repository root, for example:
python -m scripts.plan_site04 --start-xy-m -5697.5 -10002.5 \
    --goal-xy-m -4497.5 -10002.5 --slope-weight 2
These endpoints are demonstration samples, not operational EVA locations.
With --illumination, analyze a bounded UTC window and compare local-shadow costs;
see README.md for the required window, bounds, and shadow-weight arguments.
"""

import argparse
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.cost import build_cost_surface
from core.plan import find_route, route_statistics
from core.terrain import compute_slope_deg, load_dem_m


def _observer_cells(bounds_m, transform_m, shape):
    """Snap requested (west,south,east,north) meters outward to native cells."""
    if len(bounds_m) != 4 or not np.isfinite(bounds_m).all():
        raise ValueError("Output bounds must contain four finite meter coordinates")
    west, south, east, north = bounds_m
    if west >= east or south >= north:
        raise ValueError("Output bounds must have positive width and height")
    window = from_bounds(*bounds_m, transform=transform_m)
    r0, c0 = int(np.floor(window.row_off)), int(np.floor(window.col_off))
    r1 = int(np.ceil(window.row_off + window.height))
    c1 = int(np.ceil(window.col_off + window.width))
    if r0 < 1 or c0 < 1 or r1 > shape[0]-1 or c1 > shape[1]-1:
        raise ValueError("Output bounds must leave an interior DEM neighborhood")
    rows, columns = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    cells = np.column_stack((rows.ravel(), columns.ravel()))
    return cells, (slice(r0, r1), slice(c0, c1)), transform_m * Affine.translation(c0, r0)


def _solar_for_cells(cells, elevation_m, transform_m, sun_m, radius_m):
    from core.illum import solar_angles_deg

    rows, columns = cells.T
    return solar_angles_deg(
        transform_m.c + (columns+0.5)*transform_m.a,
        transform_m.f + (rows+0.5)*transform_m.e,
        elevation_m[rows, columns], sun_m, radius_m=radius_m,
    )


def _scan_horizons(elevation_m, transform_m, cells, nodes_deg, distance_step_m, radius_m):
    """One simple chunk loop; no persistent horizon cache."""
    from core.illum import terrain_horizons_deg

    horizon_deg = np.empty_like(nodes_deg)
    distance_m = np.empty_like(nodes_deg)
    samples = 0
    started_seconds = perf_counter()
    for first in range(0, len(cells), 4096):
        part = slice(first, first+4096)
        result = terrain_horizons_deg(
            elevation_m, transform_m, cells[part], nodes_deg[part],
            distance_step_m=distance_step_m, radius_m=radius_m,
        )
        horizon_deg[part] = result["horizon_deg"]
        distance_m[part] = result["horizon_distance_m"]
        samples += result["sample_count"]
        if first % (4096*8) == 0:
            print(f"Horizon progress: {min(first+4096, len(cells))}/{len(cells)} observers",
                  flush=True)
    return horizon_deg, distance_m, {
        "seconds": perf_counter()-started_seconds, "terrain_samples": samples,
        "distance_step_m": distance_step_m, "radius_m": radius_m,
    }


def _window_fraction(cells, elevation_m, transform_m, sun_m, times_seconds,
                     nodes_deg, horizon_deg, radius_m):
    from core.illum import interpolate_horizons_deg, mean_shadow_fraction

    fraction = np.empty(len(cells))
    for first in range(0, len(cells), 4096):
        part = slice(first, first+4096)
        azimuth_deg, solar_deg = _solar_for_cells(
            cells[part], elevation_m, transform_m, sun_m, radius_m)
        # Each radius has slightly different topocentric directions. The primary
        # nodes span all three radius cases so no sensitivity run extrapolates.
        horizon_at_sun_deg = interpolate_horizons_deg(
            nodes_deg[part], horizon_deg[part], azimuth_deg)
        fraction[part] = mean_shadow_fraction(solar_deg, horizon_at_sun_deg, times_seconds)
    return fraction


def illuminate_site04(
    data_dir: Path, kernel_dir: Path, output_png: Path, start_xy_m: tuple[float, float],
    goal_xy_m: tuple[float, float], slope_weight: float, shadow_weight: float,
    start_utc: str, end_utc: str, bounds_m: tuple[float, float, float, float],
    time_step_seconds: float,
) -> None:
    """Benchmark and analyze local shadow; outputs in data/ are never committed.

Bounds/positions are projected meters; time step is seconds; preference weights
are dimensionless analyst choices. Radius and numerical checks follow DECISIONS.md.
"""
    import spiceypy as spice

    from core.illum import (
        FRAME,
        KERNEL_NAMES,
        RADIUS_M,
        azimuth_nodes_deg,
        interpolate_horizons_deg,
        sample_times_seconds,
        spice_kernels,
        sun_positions_m,
        surface_basis,
        terrain_horizons_deg,
    )

    if not np.isfinite(shadow_weight) or shadow_weight <= 0:
        raise ValueError("Illumination demo requires a finite positive shadow_weight")
    analysis_started_seconds = perf_counter()
    elevation_m, transform_m, crs = load_dem_m(data_dir / "Site04_final_adj_5mpp_surf.tif")
    parameters = crs.to_dict()
    if (parameters.get("proj") != "stere" or parameters.get("lat_0") != -90
            or parameters.get("lon_0", 0) != 0 or parameters.get("R") != RADIUS_M
            or parameters.get("lat_ts", -90) != -90
            or parameters.get("x_0", 0) != 0 or parameters.get("y_0", 0) != 0):
        raise ValueError("Illumination requires the verified Site04 spherical polar CRS")
    cells, slices, output_transform_m = _observer_cells(bounds_m, transform_m, elevation_m.shape)
    for name, point_m in (("start", start_xy_m), ("goal", goal_xy_m)):
        if np.shape(point_m) != (2,) or not np.isfinite(point_m).all():
            raise ValueError(f"{name} must contain finite projected meter coordinates")
        row, col = rasterio.transform.rowcol(transform_m, *point_m)
        if not (slices[0].start <= row < slices[0].stop
                and slices[1].start <= col < slices[1].stop):
            raise ValueError(f"{name} must lie inside output bounds")
    if not np.isfinite(elevation_m[slices]).all():
        raise ValueError("Output contains invalid observer elevations; choose valid bounds")
    midpoint_xy_m = (np.asarray(start_xy_m)+np.asarray(goal_xy_m))/2
    representative = np.array([rasterio.transform.rowcol(transform_m, *midpoint_xy_m)])
    r0, c0 = (int(np.clip(s.start+(s.stop-s.start)//2-32, 1, size-65))
              for s, size in zip(slices, elevation_m.shape))
    rr, cc = np.meshgrid(np.arange(r0, r0+64), np.arange(c0, c0+64), indexing="ij")
    benchmark_cells = np.column_stack((rr.ravel(), cc.ravel()))
    report = {"start_utc": start_utc, "end_utc": end_utc, "frame": FRAME,
              "shadow_weight": shadow_weight, "slope_weight": slope_weight,
              "time_step_seconds": time_step_seconds, "kernel_sha256": {},
              "spiceypy_version": spice.__version__, "toolkit_version": spice.tkvrsn("TOOLKIT")}
    with spice_kernels(kernel_dir):
        coarse_times = sample_times_seconds(start_utc, end_utc, time_step_seconds)
        fine_times = sample_times_seconds(start_utc, end_utc, time_step_seconds/2)
        sun_coarse_m, sun_fine_m = sun_positions_m(coarse_times), sun_positions_m(fine_times)
        midpoint_sun_m = sun_positions_m([(coarse_times[0]+coarse_times[-1])/2])
        # Independent surface-observer aberration check at corners and center.
        check_cells = np.vstack((cells[0], cells[-1], cells[slices[1].stop-slices[1].start-1],
                                 cells[-(slices[1].stop-slices[1].start)], representative))
        angular_errors_deg = []
        for row, col in check_cells:
            x_m, y_m = rasterio.transform.xy(transform_m, int(row), int(col))
            up = surface_basis(x_m, y_m)[0]
            position_m = (RADIUS_M+elevation_m[row, col])*up
            for index in (0, len(coarse_times)//2, len(coarse_times)-1):
                direct = spice.spkcpo("SUN", float(coarse_times[index]), FRAME, "OBSERVER",
                                      "CN+S", position_m/1000, "MOON", FRAME)[0][:3]
                translated = sun_coarse_m[index]-position_m
                a = direct/np.linalg.norm(direct)
                b = translated/np.linalg.norm(translated)
                angular_errors_deg.append(float(np.rad2deg(np.arctan2(
                    np.linalg.norm(np.cross(a, b)), np.dot(a, b)))))
        report["surface_observer_max_error_deg"] = max(angular_errors_deg)
    for name in KERNEL_NAMES:
        report["kernel_sha256"][name] = (kernel_dir / (name+".sha256")).read_text().split()[0]
    with (data_dir / "Site04_final_adj_5mpp_surf.tif").open("rb") as stream:
        report["dem_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()

    az, _ = _solar_for_cells(benchmark_cells, elevation_m, transform_m, sun_fine_m, RADIUS_M)
    benchmark_nodes = az.min(1)[:, None] + np.ptp(az, axis=1)[:, None]*np.linspace(0, 1, 8)
    started_seconds = perf_counter()
    benchmark = terrain_horizons_deg(elevation_m, transform_m, benchmark_cells, benchmark_nodes)
    benchmark_seconds = perf_counter()-started_seconds
    throughput = benchmark["sample_count"]/benchmark_seconds
    report["benchmark"] = {"observers": 4096, "nodes": 8, "seconds": benchmark_seconds,
                           "terrain_samples": benchmark["sample_count"],
                           "samples_per_second": throughput}
    # Conservative work bound uses the whole tile diagonal, not the central
    # chunk's shorter rays. South-polar stereographic scale is >=1, so this
    # projected diagonal also bounds reference-sphere arc distances here.
    diagonal_m = np.hypot(elevation_m.shape[0]*abs(transform_m.e),
                         elevation_m.shape[1]*transform_m.a)
    maximum_steps = int(np.ceil(diagonal_m/5))+1
    # Resize before scanning any full output. Preserve endpoints in both axes.
    node_count = azimuth_nodes_deg(az).shape[1]
    estimated_seconds = len(cells)*node_count*maximum_steps/throughput
    if estimated_seconds > 1800:
        capacity = int(1800*throughput/(node_count*maximum_steps))
        width = slices[1].stop-slices[1].start
        height = min(slices[0].stop-slices[0].start, capacity//width)
        endpoint_rc = np.array([rasterio.transform.rowcol(transform_m, *p)
                                for p in (start_xy_m, goal_xy_m)])
        center_r = (slices[0].start+slices[0].stop)//2
        center_c = (slices[1].start+slices[1].stop)//2
        min_height = 2*int(np.max(np.abs(endpoint_rc[:, 0]-center_r)))+2
        min_width = 2*int(np.max(np.abs(endpoint_rc[:, 1]-center_c)))+2
        if height < min_height:
            height = min_height
            width = min(width, capacity//height)
        if width < min_width or height < 2:
            raise RuntimeError("No endpoint-containing output fits the 30-minute estimate")
        top, left = center_r-height//2, center_c-width//2
        bounds_m = array_bounds(height, width, transform_m*Affine.translation(left, top))
        cells, slices, output_transform_m = _observer_cells(bounds_m, transform_m, elevation_m.shape)

    # First full-area diagnostic: direct midpoint rays, no azimuth interpolation.
    midpoint_az = np.empty((len(cells), 1))
    midpoint_el = np.empty_like(midpoint_az)
    for first in range(0, len(cells), 4096):
        part = slice(first, first+4096)
        midpoint_az[part], midpoint_el[part] = _solar_for_cells(
            cells[part], elevation_m, transform_m, midpoint_sun_m, RADIUS_M)
    midpoint_horizon, _, midpoint_timing = _scan_horizons(
        elevation_m, transform_m, cells, midpoint_az, 5, RADIUS_M)
    known = np.isfinite(midpoint_horizon)
    clear = (midpoint_el > midpoint_horizon) & known
    if not known.any():
        raise RuntimeError("No known midpoint profiles; cannot report a boundary-exit fraction")
    boundary_percent = float(100*clear.sum()/known.sum())
    report["boundary_exit"] = {"percent": boundary_percent, "known": int(known.sum()),
                               "unknown": int((~known).sum()), **midpoint_timing}
    print(f"BOUNDARY-EXIT FRACTION: {boundary_percent:.6f}% (midpoint)", flush=True)

    az_one, el_one = _solar_for_cells(representative, elevation_m, transform_m, sun_fine_m, RADIUS_M)
    nodes_one = azimuth_nodes_deg(az_one)
    dense_nodes = np.append(np.arange(az_one.min(), az_one.max(), 0.05), az_one.max())[None, :]
    dense = terrain_horizons_deg(elevation_m, transform_m, representative, dense_nodes)
    sparse = terrain_horizons_deg(elevation_m, transform_m, representative, nodes_one)
    error_deg = float(np.max(np.abs(dense["horizon_deg"]-interpolate_horizons_deg(
        nodes_one, sparse["horizon_deg"], dense_nodes))))
    report["interpolation"] = {"max_error_deg": error_deg,
                               "representative_rc": representative[0].tolist(),
                               "azimuth_range_deg": [float(az_one.min()), float(az_one.max())],
                               "solar_elevation_range_deg": [float(el_one.min()), float(el_one.max())]}
    print(f"AZIMUTH INTERPOLATION MAX ERROR: {error_deg:.9f} deg", flush=True)
    print("BENCHMARK: " + json.dumps(report["benchmark"]), flush=True)

    # Span the actual directions of both time samplings and all radius cases.
    # This adds no angular nodes unless the measured interval requires them.
    low = np.full(len(cells), np.inf)
    high = np.full(len(cells), -np.inf)
    solar_min, solar_max = np.inf, -np.inf
    for radius in (RADIUS_M, RADIUS_M-1000, RADIUS_M+1000):
        for first in range(0, len(cells), 4096):
            part = slice(first, first+4096)
            az, el = _solar_for_cells(cells[part], elevation_m, transform_m, sun_fine_m, radius)
            low[part] = np.minimum(low[part], az.min(1))
            high[part] = np.maximum(high[part], az.max(1))
            solar_min, solar_max = min(solar_min, el.min()), max(solar_max, el.max())
    nodes = azimuth_nodes_deg(np.column_stack((low, high)))
    report["max_azimuth_sweep_deg"] = float(np.max(high-low))
    report["solar_elevation_range_deg"] = [float(solar_min), float(solar_max)]
    report["nodes"] = nodes.shape[1]
    report["estimated_horizon_seconds"] = len(cells)*nodes.shape[1]*maximum_steps/throughput
    report["representative_estimate_seconds"] = (
        benchmark_seconds*len(cells)/4096*nodes.shape[1]/8)
    print("WORK ESTIMATE: " + json.dumps({k: report[k] for k in (
        "nodes", "max_azimuth_sweep_deg", "estimated_horizon_seconds",
        "representative_estimate_seconds")}), flush=True)
    if report["estimated_horizon_seconds"] > 1800:
        raise RuntimeError("Actual node count exceeds runtime budget; reduce --bounds-m")

    horizon, distance, nominal_timing = _scan_horizons(
        elevation_m, transform_m, cells, nodes, 5, RADIUS_M)
    fraction = _window_fraction(cells, elevation_m, transform_m, sun_coarse_m, coarse_times,
                                nodes, horizon, RADIUS_M)
    if not np.isfinite(fraction).any():
        raise RuntimeError("All window illumination is unknown; cannot produce a usable raster")
    temporal_fraction = _window_fraction(cells, elevation_m, transform_m, sun_fine_m, fine_times,
                                         nodes, horizon, RADIUS_M)
    report["nominal"] = nominal_timing
    report["temporal_max_fraction_change"] = float(np.nanmax(np.abs(fraction-temporal_fraction)))
    report["temporal_mean_fraction_change"] = float(np.nanmean(np.abs(fraction-temporal_fraction)))
    variant_fractions = {"temporal": temporal_fraction}
    report["radius_sensitivity"] = []
    for radius in (RADIUS_M-1000, RADIUS_M+1000):
        variant, variant_distance, timing = _scan_horizons(
            elevation_m, transform_m, cells, nodes, 5, radius)
        variant_fraction = _window_fraction(cells, elevation_m, transform_m, sun_coarse_m,
                                            coarse_times, nodes, variant, radius)
        difference_deg = variant-horizon
        # Translate angle changes to equivalent vertical differences at the
        # farther horizon-setting distance; use one pixel for the local tangent.
        baseline_m = np.maximum(np.maximum(distance, variant_distance), 5)
        equivalent_m = np.abs(np.tan(np.deg2rad(variant))-np.tan(np.deg2rad(horizon))) * baseline_m
        mean_change = float(np.nanmean(np.abs(variant_fraction-fraction)))
        result = {**timing, "max_angle_change_deg": float(np.nanmax(np.abs(difference_deg))),
                  "rms_angle_change_deg": float(np.sqrt(np.nanmean(difference_deg**2))),
                  "max_equivalent_height_change_m": float(np.nanmax(equivalent_m)),
                  "exceeds_0p3m_height_reference": bool(np.nanmax(equivalent_m) >= 0.3),
                  "mean_fraction_change": mean_change,
                  "max_fraction_change": float(np.nanmax(np.abs(variant_fraction-fraction)))}
        report["radius_sensitivity"].append(result)
        print("RADIUS SENSITIVITY: " + json.dumps(result), flush=True)
        # Equivalent height is a diagnostic compared with the DEM's quoted
        # 0.3--0.5 m typical RMS uncertainty. Classification is the operative
        # quantity for this raster: stop only at the agreed one-percentage-point
        # area-mean change, while reporting any height-reference exceedance.
        if mean_change >= 0.01:
            raise RuntimeError("Radius sensitivity materially changes the shadow raster")
        variant_fractions[str(int(radius))] = variant_fraction
    spatial_horizon, spatial_distance, spatial_timing = _scan_horizons(
        elevation_m, transform_m, cells, nodes, 2.5, RADIUS_M)
    spatial_fraction = _window_fraction(cells, elevation_m, transform_m, sun_coarse_m,
                                       coarse_times, nodes, spatial_horizon, RADIUS_M)
    variant_fractions["spatial"] = spatial_fraction
    worst_observer, worst_node = np.unravel_index(
        np.nanargmax(np.abs(spatial_horizon-horizon)), horizon.shape)
    report["spatial"] = {**spatial_timing,
                         "max_fraction_change": float(np.nanmax(np.abs(spatial_fraction-fraction))),
                         "mean_fraction_change": float(np.nanmean(np.abs(spatial_fraction-fraction))),
                         "max_horizon_change_deg": float(np.nanmax(np.abs(spatial_horizon-horizon))),
                         "worst_observer_rc": cells[worst_observer].tolist(),
                         "worst_azimuth_deg": float(nodes[worst_observer, worst_node]),
                         "worst_horizon_5m_deg": float(horizon[worst_observer, worst_node]),
                         "worst_horizon_2p5m_deg": float(spatial_horizon[worst_observer, worst_node]),
                         "worst_distance_5m_m": float(distance[worst_observer, worst_node]),
                         "worst_distance_2p5m_m": float(spatial_distance[worst_observer, worst_node])}

    shape = tuple(s.stop-s.start for s in slices)
    bounds = array_bounds(*shape, output_transform_m)
    report["output_bounds_m"] = list(bounds)
    report["output_shape"] = list(shape)
    report["mean_shadow_fraction"] = float(np.nanmean(fraction))
    report["unknown_pixels"] = int(np.isnan(fraction).sum())
    raster = fraction.reshape(shape)
    raster_path = data_dir / "site04_mean_local_shadow.tif"
    with rasterio.open(raster_path, "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="float64", crs=crs, transform=output_transform_m,
                       nodata=np.nan, compress="deflate") as output:
        output.write(raster, 1)
        output.update_tags(start_utc=start_utc, end_utc=end_utc, frame=FRAME,
                           coverage="Local DEM only; unobstructed does not imply globally lit",
                           radius_m=RADIUS_M, time_step_seconds=time_step_seconds,
                           distance_step_m=5, azimuth_max_spacing_deg=0.5)
    slope_deg = compute_slope_deg(elevation_m, transform_m)[slices]
    cropped_z_m = elevation_m[slices]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    artist = axes[0].imshow(np.ma.masked_invalid(raster), origin="upper", extent=extent,
                            vmin=0, vmax=1, cmap="gray_r")
    fig.colorbar(artist, ax=axes[0], label="Mean local Sun-center shadow fraction")
    axes[0].set_title(f"Midpoint boundary exits: {boundary_percent:.2f}%")
    axes[1].plot(dense_nodes[0], dense["horizon_deg"][0], label="0.05° horizon")
    axes[1].plot(nodes_one[0], sparse["horizon_deg"][0], "o--", label="≤0.5° nodes")
    axes[1].set(xlabel="Azimuth (deg)", ylabel="Horizon elevation (deg)",
                title=f"Representative interpolation error: {error_deg:.5f}°")
    axes[1].legend()
    axes[2].imshow(cropped_z_m, origin="upper", extent=extent, cmap="terrain")
    report["routes"] = []
    for weight, color in ((0.0, "cyan"), (shadow_weight, "magenta")):
        costs = build_cost_surface(slope_deg, slope_limit_deg=20, slope_weight=slope_weight,
                                   shadow_fraction=raster, shadow_weight=weight)
        route = find_route(costs, output_transform_m, start_xy_m, goal_xy_m)
        summary = {k: v for k, v in route.items() if k not in ("route_rc", "route_xy_m")}
        summary["shadow_weight"] = weight
        if route["status"] == "ok":
            path = route["route_rc"]
            summary.update(route_statistics(path, cropped_z_m, slope_deg, output_transform_m))
            rr, cc = path.T
            edge_m = np.hypot(np.diff(rr)*transform_m.e, np.diff(cc)*transform_m.a)
            exposure = raster[rr, cc]
            summary["distance_weighted_mean_local_shadow_fraction"] = float(
                np.sum(edge_m*(exposure[:-1]+exposure[1:])/2)/edge_m.sum()
                if len(path) > 1 else exposure[0])
            axes[0].plot(*route["route_xy_m"].T, color=color, label=f"Shadow weight {weight:g}")
            axes[2].plot(*route["route_xy_m"].T, color=color, label=f"Shadow weight {weight:g}")
            if weight > 0:
                summary["sensitivity_routes"] = {}
                for name, alternative in variant_fractions.items():
                    alternative_cost = build_cost_surface(
                        slope_deg, slope_limit_deg=20, slope_weight=slope_weight,
                        shadow_fraction=alternative.reshape(shape), shadow_weight=weight)
                    changed = find_route(alternative_cost, output_transform_m, start_xy_m, goal_xy_m)
                    summary["sensitivity_routes"][name] = {
                        "status": changed["status"],
                        "same_ordered_cells": bool(np.array_equal(path, changed["route_rc"]))}
        report["routes"].append(summary)
    for ax in (axes[0], axes[2]):
        ax.set(xlabel="Projected X (m)", ylabel="Projected Y (m)")
        ax.scatter(*start_xy_m, marker="o", color="red")
        ax.scatter(*goal_xy_m, marker="*", color="red")
        ax.legend()
    axes[2].set_title("20° planning comparison; weighted meters, not energy")
    fig.suptitle(f"Site04 {start_utc} — {end_utc}\nLocal terrain only; no global sunlight guarantee")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    report["total_analysis_seconds"] = perf_counter()-analysis_started_seconds
    # Linux ru_maxrss is KiB; this CLI targets the project's Linux environment.
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    (data_dir / "site04_illumination_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("FINAL REPORT: " + json.dumps(report, allow_nan=False), flush=True)


def plan_site04(
    data_dir: Path,
    output_png: Path,
    start_xy_m: tuple[float, float],
    goal_xy_m: tuple[float, float],
    slope_weight: float,
) -> None:
    """Print route summaries and save maps/profiles; positions and heights are meters.

    slope_weight is a required dimensionless preference, not a metabolic constant.
    The 20-degree case is a literature-informed planning comparison; 15 and 25
    degrees are analyst-selected sensitivity cases, not an operational range.
    See DECISIONS.md for sources and limits of the milestone-2a assessment.
    No terrain is cropped for routing or statistics; map axes zoom for inspection.
    """
    started_seconds = perf_counter()
    dem_path = data_dir / "Site04_final_adj_5mpp_surf.tif"
    elevation_m, transform_m, _ = load_dem_m(dem_path)
    with dem_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    slope_deg = compute_slope_deg(elevation_m, transform_m)
    print(f"DEM SHA-256: {digest}", flush=True)
    print(f"Load/hash/slope seconds: {perf_counter() - started_seconds:.6f}", flush=True)
    print("Objective: terrain preference in weighted meters, NOT energy.", flush=True)
    print("20 deg: literature-informed comparison; 15/25 deg: sensitivity cases.", flush=True)
    print("Cutoffs allow equality; these are not operational EVA safety limits.", flush=True)
    print("Eight-neighbor diagonals may touch one or two blocked side cells.", flush=True)

    west_m, south_m, east_m, north_m = array_bounds(*elevation_m.shape, transform_m)
    extent_m = (west_m, east_m, south_m, north_m)
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.65")
    view_points_m = [np.asarray([start_xy_m, goal_xy_m])]
    for column, slope_limit_deg in enumerate((15, 20, 25)):
        started_seconds = perf_counter()
        weights = build_cost_surface(
            slope_deg, slope_limit_deg=slope_limit_deg, slope_weight=slope_weight
        )
        cost_seconds = perf_counter() - started_seconds
        started_seconds = perf_counter()
        result = find_route(weights, transform_m, start_xy_m, goal_xy_m)
        route_seconds = perf_counter() - started_seconds
        summary = {key: value for key, value in result.items()
                   if key not in ("route_rc", "route_xy_m")}
        summary.update(slope_limit_deg=slope_limit_deg, slope_weight=slope_weight,
                       cost_surface_seconds=cost_seconds, routing_seconds=route_seconds)
        # Compare route length with the Euclidean separation of its snapped
        # pixel-center endpoints, in the same projected-meter convention.
        start_center_m = np.asarray(result["snapped_start_xy_m"])
        goal_center_m = np.asarray(result["snapped_goal_xy_m"])
        separation_m = goal_center_m - start_center_m
        straight_length_m = float(np.hypot(*separation_m))
        summary["straight_line_distance_m"] = straight_length_m
        axis, profile = axes[:, column]
        artist = axis.imshow(
            np.ma.masked_where(~np.isfinite(weights), slope_deg),
            extent=extent_m, origin="upper", cmap=cmap, vmin=0, vmax=25,
        )
        axis.plot([start_center_m[0], goal_center_m[0]],
                  [start_center_m[1], goal_center_m[1]], "w--", linewidth=1,
                  label=f"Direct: {straight_length_m:.0f} m")
        axis.scatter(*start_xy_m, marker="o", color="white", edgecolor="black", label="Start")
        axis.scatter(*goal_xy_m, marker="*", color="red", edgecolor="black", label="Goal")
        axis.set(xlabel="Projected X (m)", ylabel="Projected Y (m)")
        profile.set(xlabel="Cumulative projected distance (m)", ylabel="DEM elevation (m)")
        if result["status"] == "ok":
            route_rc = result["route_rc"]
            route_xy_m = result["route_xy_m"]
            stats = route_statistics(route_rc, elevation_m, slope_deg, transform_m)
            summary.update(stats)
            summary["route_cell_count"] = len(route_rc)
            # Check full-resolution route constraints, independent of display rendering.
            rows, columns = route_rc.T
            if not np.all(np.isfinite(weights[rows, columns])):
                raise RuntimeError("Route visits blocked terrain")
            if stats["max_terrain_slope_deg"] > slope_limit_deg:
                raise RuntimeError("Route violates terrain-slope cutoff")
            axis.plot(route_xy_m[:, 0], route_xy_m[:, 1], color="orangered", linewidth=1)
            view_points_m.append(route_xy_m)
            # Distances follow pixel centers in projected meters, not array indices.
            step_xy_m = np.diff(route_xy_m, axis=0)
            cumulative_m = np.r_[0, np.cumsum(np.hypot(step_xy_m[:, 0], step_xy_m[:, 1]))]
            profile.plot(cumulative_m, elevation_m[rows, columns], color="black", linewidth=1)
            axis.set_title(
                f"Cutoff {slope_limit_deg}° | {stats['projected_length_m']:.1f} m\n"
                f"Corner steps: {result['diagonal_blocked_side_steps']} any / "
                f"{result['diagonal_both_blocked_steps']} both"
            )
            profile.set_title(
                f"Ascent {stats['ascent_m']:.1f} m | descent {stats['descent_m']:.1f} m"
            )
        else:
            axis.set_title(f"Cutoff {slope_limit_deg}° | no route")
            profile.set_axis_off()
            profile.text(0.5, 0.5, "No route", ha="center", transform=profile.transAxes)
        axis.legend(loc="upper right")
        print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
        del weights

    # A shared view encloses all routes; this display-only padding is 5% of
    # their span, with one pixel minimum for coincident endpoints.
    points_m = np.concatenate(view_points_m)
    lower_m, upper_m = points_m.min(axis=0), points_m.max(axis=0)
    padding_m = np.maximum((upper_m - lower_m) * 0.05, (abs(transform_m.a), abs(transform_m.e)))
    for axis in axes[0]:
        axis.set_xlim(lower_m[0] - padding_m[0], upper_m[0] + padding_m[0])
        axis.set_ylim(lower_m[1] - padding_m[1], upper_m[1] + padding_m[1])
    figure.colorbar(artist, ax=axes[0].tolist(), label="Terrain slope (deg); gray = blocked")
    figure.suptitle(
        f"Site04 | slope weight {slope_weight:g} | weighted meters, not energy\n"
        "20°: literature-informed comparison; 15°/25°: sensitivity cases, not safety limits"
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=150)
    plt.close(figure)
    print(f"Wrote {output_png}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/Site04"))
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--start-xy-m", type=float, nargs=2, required=True, metavar=("X_M", "Y_M"))
    parser.add_argument("--goal-xy-m", type=float, nargs=2, required=True, metavar=("X_M", "Y_M"))
    parser.add_argument("--slope-weight", type=float, required=True)
    parser.add_argument("--illumination", action="store_true")
    parser.add_argument("--kernel-dir", type=Path, default=Path("data/kernels"))
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--shadow-weight", type=float)
    parser.add_argument("--bounds-m", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--time-step-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.illumination:
        if any(v is None for v in (args.start_utc, args.end_utc, args.shadow_weight, args.bounds_m)):
            parser.error("--illumination requires --start-utc, --end-utc, --shadow-weight, --bounds-m")
        illuminate_site04(args.data_dir, args.kernel_dir,
                         args.output_png or Path("notebooks/site04_illumination.png"),
                         tuple(args.start_xy_m), tuple(args.goal_xy_m), args.slope_weight,
                         args.shadow_weight, args.start_utc, args.end_utc, tuple(args.bounds_m),
                         args.time_step_seconds)
    else:
        plan_site04(args.data_dir, args.output_png or Path("notebooks/site04_route.png"),
                    tuple(args.start_xy_m), tuple(args.goal_xy_m), args.slope_weight)
