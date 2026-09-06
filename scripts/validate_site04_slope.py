"""Characterize Central minus NASA Site04 slope; no numerical pass/fail gate.

From the repository root: uv run python -m scripts.validate_site04_slope
Print full-resolution statistics and write a diagnostic PNG, without resampling
either input for the comparison. PGDA README does not document NASA's algorithm:
https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.terrain import compute_slope_deg, load_dem_m


def validate_site04(data_dir: Path, output_png: Path) -> None:
    """Print residual statistics in degrees and plot native projected coordinates in meters."""
    dem_path = data_dir / "Site04_final_adj_5mpp_surf.tif"
    reference_path = data_dir / "Site04_final_adj_5mpp_slp.tif"
    for path in (dem_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing raster: {path}. Run python scripts/download_site04.py first."
            )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        print(f"SHA-256 {path.name}: {digest}")

    elevation_m, transform_m, crs = load_dem_m(dem_path)
    with rasterio.open(reference_path) as reference:
        if reference.count != 1:
            raise ValueError("Expected a single reference slope band")
        if (
            reference.shape != elevation_m.shape
            or reference.crs != crs
            or reference.transform != transform_m
        ):
            raise ValueError("Reference and DEM grids differ; no resampling is performed")
        if reference.units[0] not in (None, "degree", "degrees", "deg"):
            raise ValueError("Reference slope units must be degrees")
        # PGDA README supplies degree units missing from the TIFF band label.
        reference_deg = reference.read(1, masked=True, out_dtype="float64").filled(np.nan)
        reference_deg = reference_deg * reference.scales[0] + reference.offsets[0]
        bounds_m = reference.bounds

    computed_deg = compute_slope_deg(elevation_m, transform_m)
    valid_computed = np.isfinite(computed_deg)
    valid_reference = np.isfinite(reference_deg)
    valid = valid_computed & valid_reference
    if not valid.any():
        raise ValueError("No comparable pixels")
    if np.any((reference_deg[valid_reference] < 0) | (reference_deg[valid_reference] > 90)):
        raise ValueError("Reference contains slope outside 0–90 degrees")
    print("Estimator comparison on the same valid pixels (RMSE in degrees):")
    for method in ("central", "horn", "forward"):
        estimate_deg = compute_slope_deg(elevation_m, transform_m, method=method)
        if not np.array_equal(np.isfinite(estimate_deg), valid_computed):
            raise ValueError("Estimator validity masks differ")
        comparison_deg = estimate_deg[valid] - reference_deg[valid]
        print(f"  {method}: {np.sqrt(np.mean(comparison_deg**2)):.12f} deg")
    del estimate_deg, comparison_deg
    residual_deg = np.where(valid, computed_deg - reference_deg, np.nan)
    differences_deg = residual_deg[valid]
    absolute_deg = np.abs(differences_deg)
    print("Exploratory characterization: Central minus NASA (degrees); NO PASS/FAIL GATE")
    print(f"Grid: {elevation_m.shape}; transform (meters): {tuple(transform_m)}")
    print(f"CRS: {crs.to_wkt()}")
    print(f"Total pixels: {valid.size:,}; compared: {valid.sum():,}")
    border_count = 2 * elevation_m.shape[0] + 2 * elevation_m.shape[1] - 4
    print(f"Excluded border: {border_count:,}")
    print(f"Excluded interior invalid DEM neighborhoods: {(~valid_computed[1:-1, 1:-1]).sum():,}")
    print(f"Excluded reference-only invalid pixels: {(valid_computed & ~valid_reference).sum():,}")
    print(f"Bias: {differences_deg.mean():.9f} deg")
    print(f"MAE: {absolute_deg.mean():.9f} deg")
    print(f"RMSE: {np.sqrt(np.mean(differences_deg**2)):.9f} deg")
    for percentile in (50, 95, 99, 100):
        print(f"P{percentile} absolute error: {np.percentile(absolute_deg, percentile):.9f} deg")
    max_row, max_column = np.unravel_index(np.nanargmax(np.abs(residual_deg)), valid.shape)
    # Raster transform maps pixel corners; +0.5 locates the sample center.
    max_x_m, max_y_m = rasterio.transform.xy(transform_m, max_row, max_column)
    print(
        f"Maximum location: row={max_row}, column={max_column}, "
        f"x={max_x_m} m, y={max_y_m} m; residual={residual_deg[max_row, max_column]:.9f} deg"
    )

    print("Spatial blocks (4x4, row/column order, full-resolution pixels):")
    row_edges = np.linspace(0, valid.shape[0], 5, dtype=int)
    column_edges = np.linspace(0, valid.shape[1], 5, dtype=int)
    for row in range(4):
        for column in range(4):
            block_deg = residual_deg[
                row_edges[row] : row_edges[row + 1], column_edges[column] : column_edges[column + 1]
            ]
            samples_deg = block_deg[np.isfinite(block_deg)]
            if samples_deg.size:
                print(
                    f"  block {row},{column}: n={samples_deg.size}, "
                    f"bias={samples_deg.mean():.6f} deg, "
                    f"RMSE={np.sqrt(np.mean(samples_deg**2)):.6f} deg"
                )
            else:
                print(f"  block {row},{column}: no comparable pixels")
    print("By NASA slope (degrees; bins are descriptive, not routing limits):")
    for lower_deg, upper_deg in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 90)):
        selected = valid & (reference_deg >= lower_deg) & (reference_deg < upper_deg)
        if upper_deg == 90:
            selected |= valid & (reference_deg == 90)
        samples_deg = residual_deg[selected]
        if samples_deg.size:
            print(
                f"  {lower_deg}–{upper_deg}: n={samples_deg.size}, "
                f"bias={samples_deg.mean():.6f} deg, "
                f"RMSE={np.sqrt(np.mean(samples_deg**2)):.6f} deg"
            )

    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    extent_m = (bounds_m.left, bounds_m.right, bounds_m.bottom, bounds_m.top)
    # Image rendering reduces pixels to figure resolution only. Statistics above
    # always use every eligible source pixel; common limits make maps comparable.
    for axis, values_deg, title in (
        (axes[0, 0], reference_deg, "NASA reference"),
        (axes[0, 1], computed_deg, "Central, projected-meter spacing"),
    ):
        artist = axis.imshow(values_deg, extent=extent_m, origin="upper", vmin=0, vmax=90)
        axis.set_title(title)
        figure.colorbar(artist, ax=axis, label="Slope (deg)")
    display_limit_deg = max(float(np.percentile(absolute_deg, 99)), np.finfo(float).eps)
    artist = axes[0, 2].imshow(
        residual_deg,
        extent=extent_m,
        origin="upper",
        cmap="RdBu_r",
        vmin=-display_limit_deg,
        vmax=display_limit_deg,
    )
    axes[0, 2].set_title("Central − NASA (color clipped at P99 |error|)")
    figure.colorbar(artist, ax=axes[0, 2], label="Residual (deg)", extend="both")
    for axis in axes[0]:
        axis.set_xlabel("Projected X (m)")
        axis.set_ylabel("Projected Y (m)")
    axes[1, 0].hist(differences_deg, bins=150, log=True)
    axes[1, 0].set(
        xlabel="Central − NASA (deg)",
        ylabel="Pixel count (log)",
        title="Full residual distribution",
    )
    histogram = axes[1, 1].hist2d(
        reference_deg[valid], differences_deg, bins=150, norm=matplotlib.colors.LogNorm()
    )
    figure.colorbar(histogram[3], ax=axes[1, 1], label="Pixel count (log)")
    axes[1, 1].set(
        xlabel="NASA slope (deg)",
        ylabel="Central − NASA (deg)",
        title="Residual versus reference slope",
    )
    # Masked averages avoid interpreting missing rows/columns as zero bias.
    residual_masked = np.ma.masked_invalid(residual_deg)
    axes[1, 2].plot(np.arange(valid.shape[0]), residual_masked.mean(axis=1), label="Row mean")
    axes[1, 2].plot(np.arange(valid.shape[1]), residual_masked.mean(axis=0), label="Column mean")
    axes[1, 2].set(
        xlabel="Raster row / column index",
        ylabel="Mean residual (deg)",
        title="Spatial bias profiles",
    )
    axes[1, 2].legend()
    figure.suptitle("Site04 exploratory slope comparison — NASA method absent from README; no gate")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=150)
    plt.close(figure)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/Site04"))
    parser.add_argument(
        "--output-png", type=Path, default=Path("notebooks/site04_slope_validation.png")
    )
    args = parser.parse_args()
    validate_site04(args.data_dir, args.output_png)
