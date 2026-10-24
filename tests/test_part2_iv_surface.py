"""
Unit tests for Part 2: Implied Volatility Inversion (py_vollib) & Vega-Weighted Smile Smoothing.
"""

import numpy as np
from src.data_loader import _bs_call_price, clean_and_blend_otm_chain
from src.iv_surface import build_smoothed_iv_surface, invert_single_iv


def test_invert_single_iv_roundtrip():
    """Verifies that py_vollib / Brent IV inversion recovers the exact input volatility."""
    F = 580.0
    K = 550.0
    T = 45.0 / 365.0
    r = 0.045
    true_sigma = 0.215

    price = float(_bs_call_price(F, K, T, r, true_sigma))
    recovered_sigma = invert_single_iv(price, F, K, T, r)
    assert np.isclose(recovered_sigma, true_sigma, atol=1e-6)


def test_smoothed_iv_surface_eliminates_butterfly_noise():
    """Verifies that smoothing in IV space produces a non-negative 2nd derivative across the dense grid."""
    chain = clean_and_blend_otm_chain(ticker="SPY", target_dte=45, use_live=False)
    surface = build_smoothed_iv_surface(chain, num_grid_points=500)

    # 1. Verify negative equity skew (left-wing IV > ATM IV)
    left_wing_iv = surface.dense_grid_df["smoothed_iv"].iloc[0]
    assert left_wing_iv > surface.atm_iv, "Equity index options must exhibit negative put skew"

    # 2. Verify tight fit to market IVs (< 25 bps RMSE) and call prices (< 5 cents RMSE)
    assert surface.iv_rmse_bps < 25.0
    assert surface.price_rmse_cents < 5.0

    # 3. Verify that smoothing in IV space eliminates the negative 2nd derivatives!
    K_d = surface.dense_grid_df["strike"].to_numpy()
    C_d = surface.dense_grid_df["smoothed_call"].to_numpy()
    d2C = np.gradient(np.gradient(C_d, K_d), K_d)
    assert np.all(d2C[2:-2] >= 0.0), "Smoothed call curve must be convex (d^2C/dK^2 >= 0) everywhere"
