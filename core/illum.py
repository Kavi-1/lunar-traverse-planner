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
