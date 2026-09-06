"""Uncalibrated slope preference; weights are dimensionless, not metabolic rates."""

import numpy as np
import numpy.typing as npt


def build_cost_surface(
    slope_deg: npt.ArrayLike,
    *,
    slope_limit_deg: float,
    slope_weight: float,
) -> npt.NDArray[np.float64]:
    """Return dimensionless weights, with infinity for blocked cells.

    slope_limit_deg is required and strictly between 0 and 90 degrees; equality
    at the cutoff is allowed. slope_weight is a required finite, nonnegative,
    dimensionless preference parameter. Neither parameter is an EVA constant.
    See DECISIONS.md for the uncalibrated model and completed EVA assessment;
    no verified usable metabolic model was found in the sources assessed.

    Masked or nonfinite slope samples are blocked. Finite slopes outside [0, 90]
    degrees are malformed input. The returned weights multiply distance in
    projected meters to give weighted meters, not energy or travel time.
    """
    if not np.isfinite(slope_limit_deg) or not 0 < slope_limit_deg < 90:
        raise ValueError("slope_limit_deg must be finite and strictly between 0 and 90")
    if not np.isfinite(slope_weight) or slope_weight < 0:
        raise ValueError("slope_weight must be finite and nonnegative")

    values_deg = np.ma.asarray(slope_deg, dtype=np.float64).filled(np.nan)
    if values_deg.ndim != 2 or values_deg.size == 0:
        raise ValueError("slope_deg must be a nonempty 2D array")
    finite = np.isfinite(values_deg)
    if np.any(finite & ((values_deg < 0) | (values_deg > 90))):
        raise ValueError("Finite slope_deg values must be between 0 and 90")

    allowed = finite & (values_deg <= slope_limit_deg)
    weights = np.full(values_deg.shape, np.inf, dtype=np.float64)
    # tan(inclination) is gradient magnitude in m/m. Squaring is an approved
    # preference choice, not an empirical metabolic law. The baseline 1 makes
    # slope_weight=0 minimize projected distance through the allowed cells.
    gradient = np.tan(np.deg2rad(values_deg[allowed]))
    with np.errstate(over="ignore"):
        allowed_weights = 1 + slope_weight * gradient**2
    if not np.isfinite(allowed_weights).all():
        raise ValueError("slope_weight produces nonfinite traversal weights")
    weights[allowed] = allowed_weights
    return weights
