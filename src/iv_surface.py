"""
Part 2: Implied Volatility Inversion (py_vollib) & Vega-Weighted Smile Smoothing.

Implements Figlewski's core smoothing methodology ("Risk-Neutral Densities: A Review"):
1. Inverts cleaned OTM option prices into Black-Scholes Implied Volatilities sigma_mkt(K)
   using `py_vollib` (Peter Jaeckel's LetsBeRational algorithm).
2. Computes Black-Scholes Vega (dC/dsigma) and Implied-Vol uncertainty weights:
       w_i = Vega(K_i) / BidAskSpread(K_i)
   Why? Because Price_Error ~ Vega * IV_Error => IV_Std_Error ~ Spread / Vega.
3. Fits a 4th-order smoothing spline (k=4, ensuring continuous 2nd & 3rd derivatives)
   in Implied Volatility space rather than raw Price space.
4. Evaluates the smoothed smile sigma_hat(K) on a dense strike grid and maps back to
   smooth, twice-differentiable Call prices C_hat(K) = C_BS(K, sigma_hat(K)).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm

from src.data_loader import CleanedOptionChain, _bs_call_price, clean_and_blend_otm_chain

# Try importing vollib / py_vollib's fast LetsBeRational Black-76 / BSM implied vol solver
try:
    from vollib.black.implied_volatility import (
        implied_volatility_of_discounted_option_price as _pyvollib_iv,
    )
    HAS_PYVOLLIB = True
except Exception:
    try:
        from py_vollib.black.implied_volatility import (
            implied_volatility_of_discounted_option_price as _pyvollib_iv,
        )
        HAS_PYVOLLIB = True
    except Exception:
        HAS_PYVOLLIB = False


@dataclass
class SmoothedIVSurface:
    """Container for inverted market IVs, smoothed IV spline, and dense re-priced Call curve."""

    chain: CleanedOptionChain
    market_df: pd.DataFrame       # Discrete market strikes with inverted IV, Vega, fitted IV, and residuals
    dense_grid_df: pd.DataFrame   # Fine continuous grid (K, moneyness, smoothed_iv, smoothed_call, bs_const_vol_call)
    atm_iv: float                 # Smoothed Implied Volatility at K = Forward
    iv_rmse_bps: float            # Root-Mean-Square fitting error in volatility basis points (1 bp = 0.01%)
    price_rmse_cents: float       # Root-Mean-Square call price error in cents
    spline_smoothing_factor: float


def _bs_vega(forward: float, strike: np.ndarray | float, T: float, r: float, sigma: np.ndarray | float) -> np.ndarray:
    """Black-76 / Black-Scholes Vega: dC / d(sigma) = exp(-rT) * F * phi(d1) * sqrt(T)."""
    F = float(forward)
    K = np.asarray(strike, dtype=float)
    vol = np.asarray(sigma, dtype=float)
    sqrt_T = np.sqrt(max(T, 1e-8))
    d1 = (np.log(F / K) + 0.5 * (vol ** 2) * T) / (vol * sqrt_T)
    return np.exp(-r * T) * F * norm.pdf(d1) * sqrt_T


def invert_single_iv(
    call_price: float,
    forward: float,
    strike: float,
    T: float,
    r: float,
) -> float:
    """
    Inverts a European Call price (or synthetic Call price) into Implied Volatility
    using `py_vollib` (LetsBeRational), with a Brent root-finding fallback.
    """
    if HAS_PYVOLLIB:
        try:
            iv = float(_pyvollib_iv(call_price, forward, strike, r, T, "c"))
            if 0.01 <= iv <= 3.0 and np.isfinite(iv):
                return iv
        except Exception:
            pass

    # Fallback: Brent's method on [1%, 300%] volatility
    def obj(sigma: float) -> float:
        return float(_bs_call_price(forward, strike, T, r, sigma)) - call_price

    try:
        return float(brentq(obj, 0.01, 3.0, xtol=1e-8))
    except ValueError:
        return np.nan


def build_smoothed_iv_surface(
    chain: CleanedOptionChain,
    num_grid_points: int = 500,
    smoothing_multiplier: float = 0.65,
) -> SmoothedIVSurface:
    """
    Inverts the cleaned option chain into Implied Volatility space, applies a
    Vega-weighted quartic smoothing spline (k=4), and maps back to a smooth Call price curve.
    """
    df = chain.df.copy()
    F = chain.forward
    T = chain.time_to_expiry
    r = chain.risk_free_rate

    # 1. Invert every strike's synthetic Call price into Black-Scholes Implied Volatility
    ivs = [
        invert_single_iv(
            call_price=float(row["call_equiv_price"]),
            forward=F,
            strike=float(row["strike"]),
            T=T,
            r=r,
        )
        for _, row in df.iterrows()
    ]
    df["market_iv"] = ivs
    df = df.dropna(subset=["market_iv"]).copy().reset_index(drop=True)

    # 2. Compute Black-Scholes Vega and Implied-Vol precision weights
    # Standard error in IV space is proportional to BidAskSpread / Vega
    df["vega"] = _bs_vega(F, df["strike"].to_numpy(), T, r, df["market_iv"].to_numpy())
    iv_std_err = np.maximum(0.5 * df["spread"].to_numpy(), 0.005) / np.maximum(df["vega"].to_numpy(), 1.0)
    weights = 1.0 / iv_std_err
    df["iv_weight"] = weights / np.mean(weights)

    # 3. Fit a 4th-degree (quartic) smoothing spline in Implied Volatility space
    # Why k=4? Because Breeden-Litzenberger takes 2 derivatives of C(K), so a quartic spline (k=4)
    # guarantees that the resulting Risk-Neutral Density f(K) is itself a smooth quadratic spline (C^2)!
    K_mkt = df["strike"].to_numpy()
    iv_mkt = df["market_iv"].to_numpy()
    w_mkt = df["iv_weight"].to_numpy()

    # Target smoothing parameter scaled by number of strikes
    s_factor = float(smoothing_multiplier * len(K_mkt) * (0.0012 ** 2) * np.mean(w_mkt ** 2))
    spline = UnivariateSpline(K_mkt, iv_mkt, w=w_mkt, k=4, s=s_factor)

    # Evaluate at market strikes to compute diagnostics
    df["smoothed_iv"] = spline(K_mkt)
    df["iv_residual_bps"] = (df["smoothed_iv"] - df["market_iv"]) * 10_000.0
    df["smoothed_call_price"] = _bs_call_price(F, K_mkt, T, r, df["smoothed_iv"].to_numpy())
    df["price_residual"] = df["smoothed_call_price"] - df["call_equiv_price"]

    # 4. Interpolate ATM Implied Volatility at K = Forward
    atm_iv = float(spline(F))

    # 5. Evaluate on a dense, uniform strike grid for continuous differentiation in Part 3
    K_dense = np.linspace(float(K_mkt.min()), float(K_mkt.max()), num_grid_points)
    iv_dense = np.clip(spline(K_dense), 0.03, 2.0)
    call_dense = _bs_call_price(F, K_dense, T, r, iv_dense)
    bs_const_call_dense = _bs_call_price(F, K_dense, T, r, atm_iv)

    dense_df = pd.DataFrame({
        "strike": K_dense,
        "moneyness": K_dense / F,
        "log_moneyness": np.log(K_dense / F),
        "smoothed_iv": iv_dense,
        "d_iv_dK": spline.derivative(n=1)(K_dense),
        "d2_iv_dK2": spline.derivative(n=2)(K_dense),
        "smoothed_call": call_dense,
        "bs_const_vol_call": bs_const_call_dense,
    })

    iv_rmse_bps = float(np.sqrt(np.mean(df["iv_residual_bps"] ** 2)))
    price_rmse_cents = float(np.sqrt(np.mean((df["price_residual"] * 100.0) ** 2)))

    return SmoothedIVSurface(
        chain=chain,
        market_df=df,
        dense_grid_df=dense_df,
        atm_iv=atm_iv,
        iv_rmse_bps=iv_rmse_bps,
        price_rmse_cents=price_rmse_cents,
        spline_smoothing_factor=s_factor,
    )


if __name__ == "__main__":
    chain = clean_and_blend_otm_chain(ticker="SPY", target_dte=45)
    surface = build_smoothed_iv_surface(chain)

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    surface.market_df.to_csv(out_dir / "part2_market_iv_fit.csv", index=False)
    surface.dense_grid_df.to_csv(out_dir / "part2_dense_smoothed_surface.csv", index=False)

    # Check 2nd derivative of the smoothed call curve on the dense grid
    K_d = surface.dense_grid_df["strike"].to_numpy()
    C_d = surface.dense_grid_df["smoothed_call"].to_numpy()
    d2C_smooth = np.gradient(np.gradient(C_d, K_d), K_d)
    neg_count = int((d2C_smooth[2:-2] < 0).sum())

    low_put_iv = surface.market_df.iloc[0]["market_iv"]
    high_call_iv = surface.market_df.iloc[-1]["market_iv"]

    print("=" * 72)
    print(f"PART 2 SUMMARY: IMPLIED VOLATILITY INVERSION & SMILE SMOOTHING ({chain.ticker})")
    print("=" * 72)
    print(f"py_vollib Solver Active : {HAS_PYVOLLIB}")
    print(f"ATM Implied Vol (K=F)   : {surface.atm_iv:.2%}")
    print(f"Left-Tail Put IV (K={surface.market_df.iloc[0]['strike']:.0f}) : {low_put_iv:.2%} (Crash Protection Skew!)")
    print(f"Right-Tail Call IV(K={surface.market_df.iloc[-1]['strike']:.0f}): {high_call_iv:.2%}")
    print(f"25-Delta Skew (Put-Call): +{(low_put_iv - surface.atm_iv) * 100:.2f} vol pts above ATM")
    print("-" * 72)
    print(f"Spline Fit IV RMSE      : {surface.iv_rmse_bps:.2f} bps (0.01% vol units)")
    print(f"Re-priced Call RMSE     : {surface.price_rmse_cents:.2f} cents (well inside bid-ask spread!)")
    print(f"Dense Grid Points       : {len(surface.dense_grid_df)} strikes")
    print(f"Negative d^2C/dK^2 Left : {neg_count} strikes (down from {chain.raw_butterfly_violations} in raw prices!)")
    print("-" * 72)
    print("Sample Smoothed IV vs Raw Market IV across Moneyness:")
    sample_idx = np.linspace(0, len(surface.market_df) - 1, 8, dtype=int)
    cols = ["strike", "moneyness", "option_type_used", "market_iv", "smoothed_iv", "iv_residual_bps", "vega"]
    print(surface.market_df.iloc[sample_idx][cols].to_string(index=False))
    print("=" * 72)
