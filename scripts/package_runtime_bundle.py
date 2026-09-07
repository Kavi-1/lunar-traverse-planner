"""Package the fixed Site04 runtime bundle; run from the repository root."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Direct script execution adds scripts/, not the repository root, to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import SOURCE_DEM_SHA256, prepare_site
from core.clones import input_hash

source_dir = Path("data/Site04")
bundle_dir = Path("data/runtime")
nominal = prepare_site(source_dir)  # Verify full-source hashes and illumination.
dem_path = source_dir / "Site04_final_adj_5mpp_surf.tif"
source_hash = input_hash(dem_path)
if source_hash != SOURCE_DEM_SHA256:
    raise ValueError("Source DEM differs from the recorded Site04 experiment")
bundle_dir.mkdir(parents=True, exist_ok=True)

with rasterio.open(dem_path) as source:
    # Zero-based column/row offsets. Preserve a one-pixel halo for the
    # 400x400 map: +X follows columns, -Y follows rows, spacing is 5 meters.
    window = Window(579, 1999, 402, 402)
    elevation_m = source.read(1, window=window)
    profile = {
        "driver": "GTiff", "width": 402, "height": 402, "count": 1,
        "dtype": "float32", "crs": source.crs,
        "transform": source.window_transform(window),
        "nodata": source.nodata, "compress": "deflate",
    }
    with rasterio.open(bundle_dir / "site04_dem_window.tif", "w", **profile) as crop:
        crop.write(elevation_m, 1)
        if source.units[0] is not None:
            crop.set_band_unit(1, source.units[0])

names = ["site04_dem_window.tif", "site04_mean_local_shadow.tif",
         "site04_illumination_report.json"]
for name in names[1:]:
    shutil.copyfile(source_dir / name, bundle_dir / name)
manifest = {
    "source_dem_sha256": source_hash,
    "files_sha256": {
        name: input_hash(bundle_dir / name, recorded=False) for name in names
    },
}
(bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
packaged = prepare_site(bundle_dir, packaged=True)
for key in ("elevation_m", "slope_deg", "shadow_fraction", "cost_weights"):
    np.testing.assert_array_equal(packaged[key], nominal[key])
if packaged["metadata"] != nominal["metadata"] or packaged["png"] != nominal["png"]:
    raise ValueError("Packaged map differs from the original preparation")
print("Verified runtime bundle:", bundle_dir.resolve())
