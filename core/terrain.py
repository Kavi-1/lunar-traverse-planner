"""Native-grid Site04 terrain operations, independent of the API.

PGDA documents elevation in meters and projected X/Y in meters:
https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README
"""

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio import Affine
from rasterio.crs import CRS


def load_dem_m(path: Path) -> tuple[npt.NDArray[np.float64], Affine, CRS]:
    """Load a PGDA elevation raster in meters, with invalid samples set to NaN.

    Returns elevation_m, the pixel-to-projected-meter transform, and lunar CRS.
    Elevation units are documented by the PGDA README above; the Site04 TIFF
    itself has no band unit label. Preserve its CRS rather than assuming Earth.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing DEM: {path}. Run python scripts/download_site04.py first."
        )
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError("Expected a single elevation band")
        if dataset.crs is None or not dataset.crs.is_projected:
            raise ValueError("DEM must have a projected CRS with meter units")
        if dataset.crs.linear_units_factor[1] != 1.0:
            raise ValueError("DEM projected coordinates must be in meters")
        if dataset.units[0] not in (None, "m", "meter", "metre", "meters", "metres"):
            raise ValueError("DEM elevation units must be meters")
        elevation_m = dataset.read(1, masked=True, out_dtype="float64").filled(np.nan)
        elevation_m = elevation_m * dataset.scales[0] + dataset.offsets[0]
        elevation_m[~np.isfinite(elevation_m)] = np.nan
        return elevation_m, dataset.transform, dataset.crs


def compute_slope_deg(
    elevation_m: npt.ArrayLike,
    transform_m: Affine,
    *,
    method: Literal["central", "horn", "forward"] = "central",
) -> npt.NDArray[np.float64]:
    """Return slope in degrees using native projected-meter spacing.

    Default: Zevenbergen–Thorne unweighted central differences, selected by
    empirical Site04 agreement (see DECISIONS.md). Optional horn and forward
    methods retain the comparison estimators. All use the same validity mask.

    Horn (1981), Hill Shading and the Reflectance Map, Proc. IEEE 69, 14–47,
    doi:10.1109/PROC.1981.11918. This is an exploratory estimator, not a verified
    reproduction of NASA's Site04 method. No surface-distance correction is made.
    Requires an axis-aligned raster at least 3x3; borders and any neighborhood
    containing invalid elevation are NaN. No padding or gap filling is applied.
    """
    if method not in ("central", "horn", "forward"):
        raise ValueError(f"Unknown slope method: {method}")
    values_m = np.ma.asarray(elevation_m, dtype=np.float64).filled(np.nan)
    if values_m.ndim != 2 or min(values_m.shape) < 3:
        raise ValueError("Elevation must be a 2D array with at least 3 rows and columns")
    if not np.all(np.isfinite(tuple(transform_m))):
        raise ValueError("Transform must be finite")
    if transform_m.b != 0 or transform_m.d != 0:
        raise ValueError("Rotated or sheared grids are not supported")
    if transform_m.a == 0 or transform_m.e == 0:
        raise ValueError("Pixel spacing must be nonzero")

    # a b c / d e f / g h i are elevations in a 3x3 raster window.
    # X increases with column for Site04; Y decreases with row. Signed affine
    # spacings preserve that direction: e is -5 m for Site04, not +5 m.
    a_m, b_m, c_m = values_m[:-2, :-2], values_m[:-2, 1:-1], values_m[:-2, 2:]
    d_m, f_m = values_m[1:-1, :-2], values_m[1:-1, 2:]
    g_m, h_m, i_m = values_m[2:, :-2], values_m[2:, 1:-1], values_m[2:, 2:]
    if method == "central":
        # Opposite neighbors span two pixels; no diagonal weighting.
        gradient_x = (f_m - d_m) / (2 * transform_m.a)
        gradient_y = (h_m - b_m) / (2 * transform_m.e)
    elif method == "horn":
        # 8 = two-pixel baseline times the transverse weights (1+2+1).
        gradient_x = ((c_m - a_m) + 2 * (f_m - d_m) + (i_m - g_m)) / (8 * transform_m.a)
        gradient_y = ((g_m - a_m) + 2 * (h_m - b_m) + (i_m - c_m)) / (8 * transform_m.e)
    else:
        # Forward means increasing row/column, with one-pixel baselines.
        # Assign both adjacent gradients to the center without a grid shift.
        center_m = values_m[1:-1, 1:-1]
        gradient_x = (f_m - center_m) / transform_m.a
        gradient_y = (h_m - center_m) / transform_m.e

    valid = np.ones(gradient_x.shape, dtype=bool)
    for row_offset in range(3):
        for column_offset in range(3):
            valid &= np.isfinite(
                values_m[
                    row_offset : row_offset + valid.shape[0],
                    column_offset : column_offset + valid.shape[1],
                ]
            )

    # dz/dx and dz/dy are m/m. atan of gradient magnitude is inclination to
    # the projected horizontal plane; convert radians to degrees explicitly.
    interior_deg = np.rad2deg(np.arctan(np.hypot(gradient_x, gradient_y)))
    slope_deg = np.full(values_m.shape, np.nan, dtype=np.float64)
    slope_deg[1:-1, 1:-1] = np.where(valid, interior_deg, np.nan)
    return slope_deg
