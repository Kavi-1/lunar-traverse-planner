"""Run the unsourced 15/20/25-degree cutoff sweep on the full Site04 grid.

From the repository root, for example:
python -m scripts.plan_site04 --start-xy-m -6497.5 -1502.5 \
    --goal-xy-m -4497.5 -3502.5 --slope-weight 2
These endpoints are demonstration samples, not operational EVA locations.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from rasterio.transform import array_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.cost import build_cost_surface
from core.plan import find_route, route_statistics
from core.terrain import compute_slope_deg, load_dem_m


def plan_site04(
    data_dir: Path,
    output_png: Path,
    start_xy_m: tuple[float, float],
    goal_xy_m: tuple[float, float],
    slope_weight: float,
) -> None:
    """Print route summaries and save maps/profiles; positions and heights are meters.

    slope_weight is a required dimensionless preference, not a metabolic constant.
    Cutoffs are user-approved unsourced placeholders pending milestone 2a.
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
    print("Cutoffs 15/20/25 deg: unsourced placeholders pending milestone 2a.", flush=True)
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
        axis, profile = axes[:, column]
        artist = axis.imshow(
            np.ma.masked_where(~np.isfinite(weights), slope_deg),
            extent=extent_m, origin="upper", cmap=cmap, vmin=0, vmax=25,
        )
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
        "15° / 20° / 25° are unsourced placeholders; full-grid search, zoomed maps"
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=150)
    plt.close(figure)
    print(f"Wrote {output_png}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/Site04"))
    parser.add_argument("--output-png", type=Path, default=Path("notebooks/site04_route.png"))
    parser.add_argument("--start-xy-m", type=float, nargs=2, required=True, metavar=("X_M", "Y_M"))
    parser.add_argument("--goal-xy-m", type=float, nargs=2, required=True, metavar=("X_M", "Y_M"))
    parser.add_argument("--slope-weight", type=float, required=True)
    args = parser.parse_args()
    plan_site04(args.data_dir, args.output_png, tuple(args.start_xy_m), tuple(args.goal_xy_m),
                args.slope_weight)
