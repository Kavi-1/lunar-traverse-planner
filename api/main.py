"""Thin HTTP interface to the fixed Site04 demonstration; core stays API-free."""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Annotated

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from matplotlib.colors import LightSource
from matplotlib.image import imsave
from pydantic import BaseModel, ConfigDict, Field
from rasterio import Affine
from rasterio.windows import Window, from_bounds

from core.clones import BOUNDS_M, END_UTC, START_UTC, input_hash, route_measurements
from core.cost import build_cost_surface
from core.plan import find_route
from core.terrain import compute_slope_deg

DATA_DIR = Path(__file__).resolve().parents[1] / "data/Site04"
# Local repeatability identity recorded in DECISIONS.md, not a NASA signature.
SOURCE_DEM_SHA256 = "38ff70dbdf2f066c2cfa94a646c3a59d653ce7f6b4528b15fc1b43ee397ce758"
# Fixed, uncalibrated analysis choices from milestones 3/4, not EVA constants.
SLOPE_LIMIT_DEG = 20.0
SLOPE_WEIGHT = 2.0
SHADOW_WEIGHT = 2.0


def prepare_site(data_dir: Path, *, packaged: bool = False) -> dict:
    """Prepare Site04 arrays in meters/degrees and dimensionless shadow/cost.

    Registration and the fixed experiment follow DECISIONS.md milestones 3/4.
    Missing data is an error; illumination is never recomputed by the server.
    packaged selects the cropped runtime bundle rather than the original DEM.
    """
    dem_path = data_dir / (
        "site04_dem_window.tif" if packaged else "Site04_final_adj_5mpp_surf.tif"
    )
    shadow_path = data_dir / "site04_mean_local_shadow.tif"
    report_path = data_dir / "site04_illumination_report.json"
    verification_path = (
        data_dir / "manifest.json" if packaged else dem_path.with_suffix(".tif.sha256")
    )
    for path in (dem_path, verification_path, shadow_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Follow README input preparation and packaging steps."
            )
    if packaged:
        manifest = json.loads(verification_path.read_text())
        paths = (dem_path, shadow_path, report_path)
        expected_hashes = manifest.get("files_sha256")
        if (manifest.get("source_dem_sha256") != SOURCE_DEM_SHA256
                or not isinstance(expected_hashes, dict)
                or set(expected_hashes) != {path.name for path in paths}):
            raise ValueError("Runtime manifest does not identify the fixed Site04 inputs")
        for path in paths:
            if input_hash(path, recorded=False) != expected_hashes[path.name]:
                raise ValueError(f"Hash mismatch: {path}")
        dem_sha256 = manifest["source_dem_sha256"]
    else:
        dem_sha256 = input_hash(dem_path)
    report = json.loads(report_path.read_text())
    expected = {
        "start_utc": START_UTC, "end_utc": END_UTC, "output_shape": [400, 400],
        "output_bounds_m": list(BOUNDS_M), "dem_sha256": dem_sha256,
        "slope_weight": SLOPE_WEIGHT, "shadow_weight": SHADOW_WEIGHT,
        "frame": "MOON_ME_DE421",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Illumination report does not match the fixed Site04 experiment")
    expected_shape = (402, 402) if packaged else (3200, 3200)
    # Pixel corners, projected meters: columns increase X, rows decrease Y.
    # The crop starts at source (row 1999, col 579), one pixel outside the map.
    expected_transform_m = (
        Affine(5, 0, -6105, 0, -5, -8995) if packaged
        else Affine(5, 0, -9000, 0, -5, 1000)
    )
    with rasterio.open(dem_path) as source:
        if (source.crs is None or not source.crs.is_projected
                or source.crs.linear_units_factor[1] != 1.0
                or source.count != 1 or source.shape != expected_shape
                or source.transform != expected_transform_m
                or source.scales != (1.0,) or source.offsets != (0.0,)
                or source.dtypes != ("float32",)
                or source.units[0] not in (None, "m", "meter", "metre", "meters", "metres")
                or source.nodata is None or not np.isnan(source.nodata)):
            raise ValueError("Unexpected nominal Site04 registration or elevation metadata")
        crs = source.crs
        window = from_bounds(*BOUNDS_M, source.transform)
        # Rasterio transforms locate corners. A one-cell halo supplies the
        # opposite neighbors even at the 400x400 output boundary; no padding.
        halo = Window(window.col_off - 1, window.row_off - 1, 402, 402)
        halo_m = source.read(1, window=halo, masked=True, out_dtype="float64").filled(np.nan)
        transform_m = source.window_transform(window)
    if halo_m.shape != (402, 402) or not np.isfinite(halo_m).all():
        raise ValueError("Site04 analysis window needs a complete valid terrain halo")
    elevation_m = halo_m[1:-1, 1:-1].copy()
    slope_deg = compute_slope_deg(halo_m, transform_m)[1:-1, 1:-1].copy()

    with rasterio.open(shadow_path) as source:
        tags = source.tags()
        if (source.count != 1 or source.shape != (400, 400) or source.crs != crs
                or source.transform != transform_m or source.scales != (1.0,)
                or source.offsets != (0.0,) or tags.get("start_utc") != START_UTC
                or tags.get("end_utc") != END_UTC or tags.get("frame") != "MOON_ME_DE421"):
            raise ValueError("Saved illumination metadata does not match experiment")
        shadow_fraction = source.read(1, masked=True, out_dtype="float64").filled(np.nan)
    for key, expected_value in (
        ("time_step_seconds", report["time_step_seconds"]),
        ("distance_step_m", report["nominal"]["distance_step_m"]),
        ("radius_m", report["nominal"]["radius_m"]),
    ):
        if float(tags.get(key, "nan")) != expected_value:
            raise ValueError(f"Illumination sampling metadata mismatch: {key}")
    if (not np.isfinite(shadow_fraction).all()
            or np.any((shadow_fraction < 0) | (shadow_fraction > 1))
            or report["unknown_pixels"] != 0
            or not np.isclose(shadow_fraction.mean(), report["mean_shadow_fraction"],
                              rtol=0, atol=1e-12)):
        raise ValueError("Saved illumination pixels disagree with the experiment")
    cost_weights = build_cost_surface(
        slope_deg, slope_limit_deg=SLOPE_LIMIT_DEG, slope_weight=SLOPE_WEIGHT,
        shadow_fraction=shadow_fraction, shadow_weight=SHADOW_WEIGHT,
    )
    # LightSource internally negates dy for top-down raster rows. Pass positive
    # 5 m spacing, not transform_m.e=-5. The light is illustrative, not SPICE:
    # 315 degrees clockwise from grid +Y, 45 degrees above the horizontal.
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LightSource.html
    hillshade = LightSource(azdeg=315, altdeg=45).hillshade(
        halo_m, vert_exag=1, dx=transform_m.a, dy=-transform_m.e,
    )[1:-1, 1:-1]
    png = BytesIO()
    imsave(png, hillshade, cmap="gray", vmin=0, vmax=1, format="png", origin="upper")
    # Shared arrays are immutable; find_route makes its own working cost copy.
    for values in (elevation_m, slope_deg, shadow_fraction, cost_weights):
        values.setflags(write=False)
    return {
        "elevation_m": elevation_m, "slope_deg": slope_deg,
        "shadow_fraction": shadow_fraction, "cost_weights": cost_weights,
        "transform_m": transform_m, "png": png.getvalue(),
        "metadata": {
            "site": "Site04", "hillshade_url": "/api/hillshade.png",
            "width_px": 400, "height_px": 400,
            "bounds_m": list(BOUNDS_M),
            "pixel_dx_m": transform_m.a, "pixel_dy_m": transform_m.e,
            "crs_wkt": crs.to_wkt(), "frame": "MOON_ME_DE421",
            "slope_limit_deg": SLOPE_LIMIT_DEG, "slope_weight": SLOPE_WEIGHT,
            "shadow_weight": SHADOW_WEIGHT, "start_utc": START_UTC, "end_utc": END_UTC,
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    started_seconds = perf_counter()
    bundle_dir = os.environ.get("LUNAR_DATA_DIR")
    if bundle_dir is not None:
        if not bundle_dir.strip():
            raise ValueError("LUNAR_DATA_DIR must name a runtime bundle directory")
        app.state.site = prepare_site(Path(bundle_dir), packaged=True)
    else:
        app.state.site = prepare_site(DATA_DIR)
    logging.getLogger("uvicorn.error").info(
        "Site04 preparation: %.3f seconds", perf_counter() - started_seconds,
    )
    yield
    del app.state.site


app = FastAPI(title="Lunar Traverse Planner", lifespan=lifespan)


@app.get("/api/site")
def site_metadata(request: Request) -> dict:
    """Return fixed map registration in projected meters and analysis settings."""
    return request.app.state.site["metadata"]


@app.get("/api/hillshade.png")
def hillshade_image(request: Request) -> Response:
    """Return the prepared illustrative 400x400 terrain image."""
    return Response(request.app.state.site["png"], media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


CoordinateM = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class RouteRequest(BaseModel):
    """Two projected X/Y coordinate pairs in meters; settings are server-fixed."""

    model_config = ConfigDict(extra="forbid")
    start_xy_m: tuple[CoordinateM, CoordinateM]
    goal_xy_m: tuple[CoordinateM, CoordinateM]


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    # Do not echo input values: malformed NaN/Infinity cannot be encoded in JSON.
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]})


@app.post("/api/route")
def route(body: RouteRequest, request: Request) -> dict:
    """Return the pixel-center route in projected meters with core statistics."""
    site = request.app.state.site
    try:
        result = find_route(site["cost_weights"], site["transform_m"],
                            body.start_xy_m, body.goal_xy_m)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    cells = result.pop("route_rc")
    # Statistics own these values; don't duplicate them at the top level.
    for key in ("total_cost_weighted_m", "diagonal_blocked_side_steps",
                "diagonal_both_blocked_steps"):
        result.pop(key)
    result["statistics"] = None
    if result["status"] == "ok":
        result["route_xy_m"] = result["route_xy_m"].tolist()
        result["statistics"] = route_measurements(
            cells, site["elevation_m"], site["slope_deg"], site["shadow_fraction"],
            site["cost_weights"], site["transform_m"],
        )
    return result
