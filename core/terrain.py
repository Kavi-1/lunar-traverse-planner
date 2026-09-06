"""Native-grid Site04 terrain operations, independent of the API.

PGDA documents elevation in meters and projected X/Y in meters:
https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README
"""

from pathlib import Path

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
