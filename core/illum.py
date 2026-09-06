"""Local Sun-center visibility on Site04; no claim about terrain outside the DEM.

Frame: https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README
Geometry: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkpos_c.html
Horizons: Barker et al. (2021), doi:10.1016/j.pss.2020.105119, section 5.
Our sampled, local, point-Sun calculation is not a reproduction of that model.
"""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import spiceypy as spice
from rasterio import Affine

# The TIFF's projection radius; explicitly approved as the physical reference
# radius too, subject to the +/-1000 m sensitivity check in DECISIONS.md.
RADIUS_M = 1_737_400.0
FRAME = "MOON_ME_DE421"
KERNEL_NAMES = (
    "naif0012.tls", "de421.bsp", "moon_pa_de421_1900-2050.bpc", "moon_080317.tf",
)


@contextmanager
def spice_kernels(kernel_dir: Path) -> Iterator[None]:
    """Load verified local kernels, then unload only files owned by this context."""
    loaded = []
    try:
        for name in KERNEL_NAMES:
            path = (kernel_dir / name).resolve()
            checksum = path.with_suffix(path.suffix + ".sha256")
            if not path.is_file() or not checksum.is_file():
                raise FileNotFoundError("Missing kernel/hash; run download_site04.py --kernels-only")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != checksum.read_text().split()[0]:
                raise ValueError(f"Kernel hash mismatch: {path}")
            # Respect a caller's already-loaded kernels, without clearing its pool.
            try:
                spice.kinfo(str(path))
            except spice.utils.exceptions.NotFoundError:
                spice.furnsh(str(path))
                loaded.append(str(path))
        yield
    finally:
        for path in reversed(loaded):
            spice.unload(path)


def sample_times_seconds(start_utc: str, end_utc: str, time_step_seconds: float) -> np.ndarray:
    """Return ET seconds past J2000 TDB, including both ISO UTC endpoints.

Accept naive ISO timestamps as UTC, or explicit Z/+00:00. Reject other zones
and non-ISO time-system annotations so an argument named UTC cannot mean TDB.
"""
    if not np.isfinite(time_step_seconds) or time_step_seconds <= 0:
        raise ValueError("time_step_seconds must be positive and finite")
    converted_seconds = []
    for text in (start_utc, end_utc):
        stamp = datetime.fromisoformat(text)
        if stamp.utcoffset() not in (None, timedelta(0)):
            raise ValueError("Window timestamps must be UTC")
        converted_seconds.append(spice.str2et(stamp.strftime("%Y-%m-%d %H:%M:%S.%f UTC")))
    start_seconds, end_seconds = converted_seconds
    if end_seconds <= start_seconds:
        raise ValueError("end_utc must be after start_utc")
    return np.append(np.arange(start_seconds, end_seconds, time_step_seconds), end_seconds)


def sun_positions_m(et_seconds: npt.ArrayLike) -> np.ndarray:
    """Return Moon-to-Sun apparent vectors in MOON_ME_DE421, meters, shape (T,3).

CN+S applies reception light time and stellar aberration at the lunar center.
The Moon-centered output frame is evaluated at the observer epoch. Every call
also checks actual ephemeris/orientation availability, including retarded time;
SPICE coverage errors propagate rather than silently substituting a frame.
"""
    times_seconds = np.asarray(et_seconds, dtype=float)
    if times_seconds.ndim != 1 or not len(times_seconds) or not np.isfinite(times_seconds).all():
        raise ValueError("et_seconds must be a nonempty finite vector")
    return np.array([
        spice.spkpos("SUN", float(t), FRAME, "CN+S", "MOON")[0] for t in times_seconds
    ]) * 1000.0


def surface_basis(
    x_m: npt.ArrayLike, y_m: npt.ArrayLike, *, projection_radius_m: float = RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radial up, east, north unit vectors, with final dimension 3.

Site04: spherical south polar stereographic, zero offsets/central meridian.
rho=2R*tan((latitude+pi/2)/2), east-positive longitude=atan2(x,y).
The rational inverse below avoids precision loss near the pole. At the pole
choose longitude=0: east=+Y, north=+X. Source: PROJ stere.cpp and TIFF CRS.
"""
    if not np.isfinite(projection_radius_m) or projection_radius_m <= 0:
        raise ValueError("projection_radius_m must be positive and finite")
    x_m, y_m = np.broadcast_arrays(np.asarray(x_m, float), np.asarray(y_m, float))
    if not np.isfinite(x_m).all() or not np.isfinite(y_m).all():
        raise ValueError("Projected coordinates must be finite")
    rho_squared_m2 = x_m*x_m + y_m*y_m
    denominator_m2 = rho_squared_m2 + 4*projection_radius_m**2
    up = np.stack((4*projection_radius_m*y_m/denominator_m2,
                   4*projection_radius_m*x_m/denominator_m2,
                   (rho_squared_m2 - 4*projection_radius_m**2)/denominator_m2), axis=-1)
    longitude_rad = np.arctan2(x_m, y_m)
    east = np.stack((-np.sin(longitude_rad), np.cos(longitude_rad),
                     np.zeros_like(x_m)), axis=-1)
    north = np.cross(up, east)
    return up, east, north


def solar_angles_deg(
    x_m: npt.ArrayLike, y_m: npt.ArrayLike, elevation_m: npt.ArrayLike,
    sun_positions_m: npt.ArrayLike, *, radius_m: float = RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unwrapped azimuth, elevation) in degrees, shape (observers, times).

Azimuth is clockwise from local north toward east. Subtract each observer
from the Moon-centered apparent Sun vector; this translation approximates
surface-observer aberration, checked separately with SPICE spkcpo.
"""
    x_m, y_m, z_m = np.broadcast_arrays(x_m, y_m, elevation_m)
    if x_m.ndim != 1 or not np.isfinite(z_m).all() or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Observers must be finite vectors and radius_m positive")
    sun_m = np.asarray(sun_positions_m, float)
    if sun_m.ndim != 2 or sun_m.shape[1] != 3 or not np.isfinite(sun_m).all():
        raise ValueError("sun_positions_m must have shape (times,3) and be finite")
    up, east, north = surface_basis(x_m, y_m)
    position_m = up * (radius_m + z_m)[:, None]
    direction_m = sun_m[None, :, :] - position_m[:, None, :]
    east_m = np.sum(direction_m * east[:, None, :], axis=-1)
    north_m = np.sum(direction_m * north[:, None, :], axis=-1)
    vertical_m = np.sum(direction_m * up[:, None, :], axis=-1)
    azimuth_rad = np.unwrap(np.arctan2(east_m, north_m), axis=1)
    return np.rad2deg(azimuth_rad), np.rad2deg(np.arctan2(vertical_m, np.hypot(east_m, north_m)))


def azimuth_nodes_deg(azimuth_deg: npt.ArrayLike, max_spacing_deg: float = 0.5) -> np.ndarray:
    """Return per-observer evenly spaced nodes covering an already unwrapped sweep."""
    angles_deg = np.asarray(azimuth_deg, float)
    if angles_deg.ndim != 2 or not angles_deg.size or not np.isfinite(angles_deg).all():
        raise ValueError("azimuth_deg must be a finite nonempty observer-by-time array")
    if not np.isfinite(max_spacing_deg) or max_spacing_deg <= 0:
        raise ValueError("max_spacing_deg must be positive and finite")
    low_deg, high_deg = angles_deg.min(axis=1), angles_deg.max(axis=1)
    count = max(2, int(np.ceil(np.max(high_deg-low_deg)/max_spacing_deg)) + 1)
    return low_deg[:, None] + (high_deg-low_deg)[:, None] * np.linspace(0, 1, count)


def terrain_horizons_deg(
    elevation_m: npt.ArrayLike, transform_m: Affine, observer_rc: npt.ArrayLike,
    azimuth_deg: npt.ArrayLike, *, distance_step_m: float = 5.0,
    radius_m: float = RADIUS_M,
) -> dict[str, object]:
    """Trace complete local horizons, returning degrees, distances in m, sample count.

Observers are integer sample indices; azimuth_deg has shape (observers,nodes).
The input must use Site04's unrotated +X/-Y meter grid and verified spherical
projection. Caller chunks observers. Unknown profiles have NaN horizons. A
finite horizon ends at the available DEM, NOT at a verified global horizon.
"""
    z_m = np.ma.asarray(elevation_m, dtype=float).filled(np.nan)
    cells = np.asarray(observer_rc)
    angles_deg = np.asarray(azimuth_deg, float)
    if z_m.ndim != 2 or min(z_m.shape) < 3:
        raise ValueError("Elevation must be a 2D grid at least 3x3")
    if (cells.ndim != 2 or cells.shape[1] != 2 or not len(cells)
            or not np.issubdtype(cells.dtype, np.integer)):
        raise ValueError("observer_rc must be nonempty integer row/column pairs")
    if np.any(cells < 1) or np.any(cells >= np.array(z_m.shape)-1):
        raise ValueError("Observers require an interior DEM neighborhood")
    if angles_deg.ndim != 2 or angles_deg.shape[0] != len(cells) or not angles_deg.size:
        raise ValueError("azimuth_deg must have shape (observers,nodes)")
    if (not np.isfinite(angles_deg).all() or not np.isfinite(distance_step_m)
            or distance_step_m <= 0 or not np.isfinite(radius_m) or radius_m <= 0):
        raise ValueError("Angles must be finite; distances and radius_m positive")
    if (not np.isfinite(tuple(transform_m)).all() or transform_m.b or transform_m.d
            or transform_m.a <= 0 or transform_m.e >= 0):
        raise ValueError("Expected an unrotated +X/-Y meter grid")
    rows, cols = np.repeat(cells, angles_deg.shape[1], axis=0).T
    x_m = transform_m.c + (cols+0.5)*transform_m.a
    y_m = transform_m.f + (rows+0.5)*transform_m.e
    up, east, north = surface_basis(x_m, y_m)
    azimuth_rad = np.deg2rad(angles_deg.ravel())
    forward = north*np.cos(azimuth_rad)[:, None] + east*np.sin(azimuth_rad)[:, None]
    observer_z_m = z_m[rows, cols]
    # Analytic one-sided derivative of the bilinear terrain at each observer.
    # d(projected coordinate)/d(projection-sphere arc length), m/m. Sampling
    # angles stay fixed when physical radius_m changes: otherwise a radius
    # sensitivity experiment also moves samples across bilinear terrain knots.
    denominator = 1-up[:, 2]
    dx = 2 * (
        forward[:, 1]*denominator + up[:, 1]*forward[:, 2]) / denominator**2
    dy = 2 * (
        forward[:, 0]*denominator + up[:, 0]*forward[:, 2]) / denominator**2
    column_sign = np.where(dx >= 0, 1, -1)
    row_sign = np.where(dy/transform_m.e >= 0, 1, -1)
    gx = (z_m[rows, cols+column_sign]-observer_z_m)/(column_sign*transform_m.a)
    gy = (z_m[rows+row_sign, cols]-observer_z_m)/(row_sign*transform_m.e)
    derivative = np.where(dx == 0, 0, gx*dx) + np.where(dy == 0, 0, gy*dy)
    horizon_rad = np.arctan2(derivative*RADIUS_M, radius_m+observer_z_m)
    known = np.isfinite(horizon_rad) & np.isfinite(observer_z_m)
    active = known.copy()
    horizon_distance_m = np.zeros(len(rows))
    boundary_distance_m = np.zeros(len(rows))
    sample_count = 0
    # Region is small and convex around the south pole. March until every
    # great-circle profile leaves the rectangle of interpolation support.
    distance_m = distance_step_m
    while active.any():
        indices = np.flatnonzero(active)
        arc_rad = distance_m/RADIUS_M
        sin_arc, cos_arc = np.sin(arc_rad), np.cos(arc_rad)
        direction = up[indices]*cos_arc + forward[indices]*sin_arc
        projected_x_m = 2*RADIUS_M*direction[:, 1]/(1-direction[:, 2])
        projected_y_m = 2*RADIUS_M*direction[:, 0]/(1-direction[:, 2])
        column = (projected_x_m-transform_m.c)/transform_m.a - 0.5
        row = (projected_y_m-transform_m.f)/transform_m.e - 0.5
        inside = (column >= 0) & (column < z_m.shape[1]-1) & (row >= 0) & (row < z_m.shape[0]-1)
        exited = indices[~inside]
        active[exited] = False
        boundary_distance_m[exited] = distance_m
        indices, row, column = indices[inside], row[inside], column[inside]
        r, c = np.floor(row).astype(int), np.floor(column).astype(int)
        fr, fc = row-r, column-c
        samples_m = ((1-fr)*((1-fc)*z_m[r, c] + fc*z_m[r, c+1])
                     + fr*((1-fc)*z_m[r+1, c] + fc*z_m[r+1, c+1]))
        sample_count += len(indices)
        valid = np.isfinite(samples_m)
        known[indices[~valid]] = False
        active[indices[~valid]] = False
        indices, samples_m = indices[valid], samples_m[valid]
        # delta.up=(R+z_sample)*cos(arc)-(R+z_observer).
        # cos(arc)-1=-2*sin(arc/2)^2 avoids subtracting million-meter radii.
        vertical_m = (samples_m-observer_z_m[indices]
                      - 2*(radius_m+samples_m)*np.sin(arc_rad/2)**2)
        horizontal_m = (radius_m+samples_m)*sin_arc
        candidate_rad = np.arctan2(vertical_m, horizontal_m)
        higher = candidate_rad > horizon_rad[indices]
        selected = indices[higher]
        horizon_rad[selected] = candidate_rad[higher]
        horizon_distance_m[selected] = distance_m
        distance_m += distance_step_m
    horizon_rad[~known] = np.nan
    shape = angles_deg.shape
    return {"horizon_deg": np.rad2deg(horizon_rad).reshape(shape),
            "horizon_distance_m": horizon_distance_m.reshape(shape),
            "boundary_distance_m": boundary_distance_m.reshape(shape),
            "sample_count": sample_count}


def interpolate_horizons_deg(
    nodes_deg: npt.ArrayLike, horizon_deg: npt.ArrayLike, azimuth_deg: npt.ArrayLike,
) -> np.ndarray:
    """Linearly interpolate per-observer horizons in degrees; never extrapolate."""
    nodes = np.asarray(nodes_deg, float)
    horizons = np.asarray(horizon_deg, float)
    angles = np.asarray(azimuth_deg, float)
    if (nodes.ndim != 2 or nodes.shape != horizons.shape or nodes.shape[1] < 2
            or angles.ndim != 2 or len(nodes) != len(angles)):
        raise ValueError("Expected matching observer-by-node arrays and observer-by-time angles")
    if (not np.isfinite(nodes).all() or not np.isfinite(angles).all()
            or np.any(np.diff(nodes, axis=1) < 0)
            or np.any(angles < nodes[:, :1]) or np.any(angles > nodes[:, -1:])):
        raise ValueError("Nodes must be ordered and enclose finite angles; no extrapolation")
    # A loop over observers is inexpensive compared with terrain scanning and
    # avoids an observer-by-node-by-time temporary. NaN brackets stay unknown.
    return np.array([np.interp(a, n, h) for a, n, h in zip(angles, nodes, horizons)])


def mean_shadow_fraction(
    solar_elevation_deg: npt.ArrayLike, horizon_deg: npt.ArrayLike, et_seconds: npt.ArrayLike,
) -> np.ndarray:
    """Time-weighted Sun-center shadow fraction (dimensionless), NaN if unknown.

Trapezoidal integration includes actual interval lengths; it does not certify
visibility between epochs or account for terrain beyond the DEM boundary.
"""
    solar, horizon = np.asarray(solar_elevation_deg, float), np.asarray(horizon_deg, float)
    times = np.asarray(et_seconds, float)
    if (solar.ndim != 2 or solar.shape != horizon.shape or times.ndim != 1
            or solar.shape[1] != len(times) or len(times) < 2
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Matching observer-by-time arrays and increasing ET seconds required")
    shadow = (solar <= horizon).astype(float)
    intervals_seconds = np.diff(times)
    fraction = np.sum((shadow[:, :-1]+shadow[:, 1:])/2*intervals_seconds, axis=1)
    fraction /= times[-1]-times[0]
    fraction[~(np.isfinite(solar) & np.isfinite(horizon)).all(axis=1)] = np.nan
    return fraction
