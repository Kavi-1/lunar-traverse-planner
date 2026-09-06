"""Cost and routing checks use actual Site04 observations, never synthetic terrain."""

from pathlib import Path

import numpy as np
import pytest

from core.cost import build_cost_surface
from core.terrain import compute_slope_deg, load_dem_m

DEM_PATH = Path(__file__).resolve().parents[1] / "data/Site04/Site04_final_adj_5mpp_surf.tif"


@pytest.fixture(scope="module")
def slope_patch_deg():
    elevation_m, transform_m, _ = load_dem_m(DEM_PATH)
    return compute_slope_deg(elevation_m[999:1005, 999:1005], transform_m)


def test_cost_real_sample_hand_checked(slope_patch_deg):
    weights = build_cost_surface(slope_patch_deg, slope_limit_deg=25, slope_weight=2)
    # Independent derivation from raw DEM neighbors of (1000,1000), in meters:
    # gx = (1236.615966796875 - 1234.658447265625)/10 = 4009/20480.
    # gy = (1236.9766845703125 - 1234.46923828125)/-10 = -20541/81920.
    # Signed two-pixel spacings are +10 m in X and -10 m in Y.
    # Exact rational arithmetic, bypassing the production atan/tan path:
    # 1 + 2*(gx*gx + gy*gy) = 4034529177/3355443200
    #                     = 1.202383392155170440673828125.
    # The original literal 1.2023771303892135 was entered incorrectly;
    # no derivation supports it, and rounding was not established as its cause.
    assert weights[1, 1] == pytest.approx(1.2023833921551703, abs=1e-12)
    assert np.isinf(weights[0]).all()
    assert weights.dtype == np.float64


def test_cutoff_equality_and_zero_preference(slope_patch_deg):
    cutoff_deg = float(slope_patch_deg[1, 1])
    original_deg = slope_patch_deg.copy()
    weights = build_cost_surface(slope_patch_deg, slope_limit_deg=cutoff_deg, slope_weight=0)
    allowed = np.isfinite(slope_patch_deg) & (slope_patch_deg <= cutoff_deg)
    assert np.all(weights[allowed] == 1)
    assert np.isinf(weights[~allowed]).all()
    assert weights[1, 1] == 1
    lower_weights = build_cost_surface(
        slope_patch_deg, slope_limit_deg=np.nextafter(cutoff_deg, 0), slope_weight=0
    )
    assert np.isinf(lower_weights[1, 1])
    np.testing.assert_array_equal(slope_patch_deg, original_deg)


def test_masked_observation_is_blocked(slope_patch_deg):
    masked_deg = np.ma.array(slope_patch_deg, mask=False)
    masked_deg.mask[1, 1] = True
    weights = build_cost_surface(masked_deg, slope_limit_deg=25, slope_weight=2)
    assert np.isinf(weights[1, 1])


@pytest.mark.parametrize("slope_limit_deg", [0, -1, 90, np.nan, np.inf])
def test_invalid_cutoff(slope_patch_deg, slope_limit_deg):
    with pytest.raises(ValueError, match="slope_limit_deg"):
        build_cost_surface(slope_patch_deg, slope_limit_deg=slope_limit_deg, slope_weight=1)


@pytest.mark.parametrize("slope_weight", [-1, np.nan, np.inf])
def test_invalid_preference(slope_patch_deg, slope_weight):
    with pytest.raises(ValueError, match="slope_weight"):
        build_cost_surface(slope_patch_deg, slope_limit_deg=25, slope_weight=slope_weight)


def test_invalid_array_shape(slope_patch_deg):
    for values_deg in (slope_patch_deg[1], slope_patch_deg[:0]):
        with pytest.raises(ValueError, match="nonempty 2D"):
            build_cost_surface(values_deg, slope_limit_deg=25, slope_weight=1)


def test_malformed_negative_slope_rejected(slope_patch_deg):
    # Negate actual observations solely to exercise invalid input handling.
    with pytest.raises(ValueError, match="Finite slope_deg"):
        build_cost_surface(-slope_patch_deg, slope_limit_deg=25, slope_weight=1)
