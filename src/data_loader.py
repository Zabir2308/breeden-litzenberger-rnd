"""
Part 1: Market Data Pipeline, Implied Forward Extraction & OTM Option Chain Cleaning.

Implements the data preparation workflow from Figlewski, "Risk-Neutral Densities: A Review":
1. Pulls real option chains (calls & puts) via yfinance (with deterministic fallback snapshot).
2. Extracts the market-implied forward price F via near-ATM Put-Call Parity:
       F = K + exp(r * T) * (C_mid(K) - P_mid(K))
3. Filters microstructure noise (zero bids, wide bid-ask spreads, intrinsic violations).
4. Blends liquid OTM Puts (K < F) and OTM Calls (K >= F) into a unified synthetic Call curve
   using Put-Call Parity:
       C_synth(K) = P_mid(K) + exp(-r * T) * (F - K)
5. Enforces static no-arbitrage bounds (-exp(-r*T) < dC/dK < 0) and flags raw butterfly
   arbitrage violations (d^2C / dK^2 < 0) to motivate Implied Volatility smoothing in Part 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class CleanedOptionChain:
    """Container for the cleaned, OTM-blended option chain and market parameters."""

    ticker: str
    expiration: str
    spot: float
    forward: float
    risk_free_rate: float
    implied_dividend_yield: float
    time_to_expiry: float  # In years (ACT/365)
    df: pd.DataFrame       # Columns: strike, option_type_used, bid, ask, mid_price, spread, call_equiv_price, ...
    raw_calls_count: int
    raw_puts_count: int
    monotonicity_dropped: int
    raw_butterfly_violations: int
    data_source: str       # 'yfinance_live' or 'calibrated_snapshot'


def _bs_call_price(forward: float, strike: np.ndarray | float, T: float, r: float, sigma: np.ndarray | float) -> np.ndarray:
    """Black-76 / Black-Scholes European call price using forward price F."""
    F = float(forward)
    K = np.asarray(strike, dtype=float)
    vol = np.asarray(sigma, dtype=float)
    sqrt_T = np.sqrt(max(T, 1e-8))
    d1 = (np.log(F / K) + 0.5 * (vol ** 2) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    disc = np.exp(-r * T)
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


def generate_realistic_spy_snapshot(
    spot: float = 580.0,
    T: float = 45.0 / 365.0,
    r: float = 0.045,
    q: float = 0.013,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, float, str]:
    """
    Generates a realistic, institutional-grade SPY option chain snapshot with equity put skew,
    discrete $1/$5 strike spacing, realistic bid-ask spreads, and tick-size microstructure noise.
    Used when offline or when Yahoo Finance rate-limits requests.
    """
    rng = np.random.default_rng(seed)
    forward = spot * np.exp((r - q) * T)

    # Strike grid from -30% to +20% around spot ($2 spacing in wings, $1 spacing near ATM)
    wings_low = np.arange(np.floor(spot * 0.72), np.floor(spot * 0.90), 2.0)
    core = np.arange(np.floor(spot * 0.90), np.ceil(spot * 1.08), 1.0)
    wings_high = np.arange(np.ceil(spot * 1.08), np.ceil(spot * 1.18) + 1.0, 2.0)
    strikes = np.unique(np.concatenate([wings_low, core, wings_high]))

    # Realistic S&P 500 / SPY Implied Volatility skew parameterization in log-moneyness k = ln(K/F)
    # ATM vol ~ 15.5%, strong negative skew (-0.38) for crash protection demand, positive curvature (smile)
    k = np.log(strikes / forward)
    atm_vol = 0.155
    skew = -0.38
    smile = 0.42
    left_wing_steepener = -0.15 * np.minimum(k, 0.0) ** 2
    true_iv = atm_vol + skew * k + smile * (k ** 2) + left_wing_steepener
    true_iv = np.clip(true_iv, 0.09, 0.65)

    # Exact call and put prices before microstructure tick noise
    disc = np.exp(-r * T)
    exact_calls = _bs_call_price(forward, strikes, T, r, true_iv)
    exact_puts = exact_calls - disc * (forward - strikes)

    # Add realistic market microstructure noise (tick rounding to $0.01 + bid-ask spread widening in ITM)
    # OTM options have tight spreads; deep ITM options have wide spreads & occasional stale quotes
    is_call_itm = strikes < forward
    call_noise = rng.normal(0.0, np.where(is_call_itm, 0.08, 0.012), size=len(strikes))
    put_noise = rng.normal(0.0, np.where(~is_call_itm, 0.08, 0.012), size=len(strikes))

    call_mid = np.maximum(exact_calls + call_noise, 0.01)
    put_mid = np.maximum(exact_puts + put_noise, 0.01)

    call_half_spread = np.where(
        is_call_itm,
        np.maximum(0.05, 0.015 * call_mid),
        np.maximum(0.01, 0.008 * call_mid + 0.005),
    )
    put_half_spread = np.where(
        ~is_call_itm,
        np.maximum(0.05, 0.015 * put_mid),
        np.maximum(0.01, 0.008 * put_mid + 0.005),
    )

    calls_df = pd.DataFrame({
        "strike": strikes,
        "bid": np.round(np.maximum(0.0, call_mid - call_half_spread), 2),
        "ask": np.round(call_mid + call_half_spread, 2),
        "volume": np.where(is_call_itm, rng.integers(5, 80, len(strikes)), rng.integers(150, 5000, len(strikes))),
        "openInterest": np.where(is_call_itm, rng.integers(50, 500, len(strikes)), rng.integers(1000, 25000, len(strikes))),
    })

    puts_df = pd.DataFrame({
        "strike": strikes,
        "bid": np.round(np.maximum(0.0, put_mid - put_half_spread), 2),
        "ask": np.round(put_mid + put_half_spread, 2),
        "volume": np.where(~is_call_itm, rng.integers(5, 80, len(strikes)), rng.integers(150, 5000, len(strikes))),
        "openInterest": np.where(~is_call_itm, rng.integers(50, 500, len(strikes)), rng.integers(1000, 25000, len(strikes))),
    })

    expiry_str = "45d-Calibrated-SPY"
    return calls_df, puts_df, spot, expiry_str


def fetch_raw_option_chain(
    ticker: str = "SPY",
    target_dte: int = 45,
    use_live: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, float, str, float, str]:
    """
    Fetches raw calls and puts DataFrames from yfinance for the expiration closest to `target_dte` days.
    Falls back cleanly to the calibrated SPY snapshot if offline or market quotes are unavailable.

    Returns:
        (calls_df, puts_df, spot_price, expiration_date, time_to_expiry_years, data_source)
    """
    if use_live:
        try:
            import socket
            with socket.create_connection(("query2.finance.yahoo.com", 443), timeout=1.5):
                pass
            import yfinance as yf

            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d")
            if not hist.empty:
                spot = float(hist["Close"].iloc[-1])
                expirations = tk.options
                if expirations:
                    now = datetime.now(timezone.utc).date()
                    exp_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in expirations]
                    dtes = [max((d - now).days, 1) for d in exp_dates]

                    # Pick expiration closest to target_dte (and at least 14 days out)
                    valid_indices = [i for i, dte in enumerate(dtes) if dte >= 14] or list(range(len(dtes)))
                    best_idx = min(valid_indices, key=lambda i: abs(dtes[i] - target_dte))
                    chosen_exp = expirations[best_idx]
                    T = dtes[best_idx] / 365.0

                    chain = tk.option_chain(chosen_exp)
                    calls_df = chain.calls.copy()
                    puts_df = chain.puts.copy()

                    # Verify that enough liquid two-sided quotes exist (during off-hours/nights,
                    # Yahoo Finance often zeroes out bids/asks or leaves wide stale quotes)
                    c_valid = calls_df[(calls_df["bid"] > 0.05) & (calls_df["ask"] > calls_df["bid"]) & ((calls_df["ask"] - calls_df["bid"]) / calls_df["ask"] < 0.25)]
                    p_valid = puts_df[(puts_df["bid"] > 0.05) & (puts_df["ask"] > puts_df["bid"]) & ((puts_df["ask"] - puts_df["bid"]) / puts_df["ask"] < 0.25)]
                    if len(c_valid) >= 25 and len(p_valid) >= 25:
                        return calls_df, puts_df, spot, chosen_exp, T, "yfinance_live"
        except Exception:
            pass

    # Fallback to deterministic institutional SPY snapshot
    T_default = target_dte / 365.0
    calls_df, puts_df, spot, chosen_exp = generate_realistic_spy_snapshot(T=T_default)
    return calls_df, puts_df, spot, chosen_exp, T_default, "calibrated_snapshot"


def estimate_implied_forward(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    spot: float,
    r: float,
    T: float,
    near_atm_pct: float = 0.04,
) -> tuple[float, float]:
    """
    Estimates the market-implied Forward price F and implied dividend yield q using Put-Call Parity
    across liquid near-the-money strikes:
        C(K) - P(K) = exp(-r * T) * (F - K)
        => F(K) = K + exp(r * T) * (C_mid(K) - P_mid(K))

    Why this matters for interviews:
    Estimating F directly from the option chain avoids dividend mis-specification and ensures
    no jump discontinuity at K = F when stitching OTM puts and OTM calls together!
    """
    c = calls_df[(calls_df["bid"] > 0) & (calls_df["ask"] > calls_df["bid"])].copy()
    p = puts_df[(puts_df["bid"] > 0) & (puts_df["ask"] > puts_df["bid"])].copy()

    c["c_mid"] = 0.5 * (c["bid"] + c["ask"])
    p["p_mid"] = 0.5 * (p["bid"] + p["ask"])

    merged = pd.merge(c[["strike", "c_mid"]], p[["strike", "p_mid"]], on="strike", how="inner")
    atm_band = merged[
        (merged["strike"] >= spot * (1.0 - near_atm_pct))
        & (merged["strike"] <= spot * (1.0 + near_atm_pct))
    ]

    if len(atm_band) >= 3:
        implied_forwards = atm_band["strike"] + np.exp(r * T) * (atm_band["c_mid"] - atm_band["p_mid"])
        forward = float(np.median(implied_forwards))
    else:
        forward = float(spot * np.exp((r - 0.013) * T))

    implied_q = r - np.log(forward / spot) / max(T, 1e-8)
    return forward, float(implied_q)


def clean_and_blend_otm_chain(
    ticker: str = "SPY",
    target_dte: int = 45,
    risk_free_rate: float = 0.045,
    max_rel_spread: float = 0.35,
    moneyness_bounds: tuple[float, float] = (0.72, 1.18),
    use_live: bool = True,
    save_csv_path: Optional[ str | Path ] = None,
) -> CleanedOptionChain:
    """
    Executes the full Part 1 pipeline:
    1. Fetches raw Call & Put chains.
    2. Computes market-implied Forward price F via near-ATM Put-Call Parity.
    3. Selects liquid OTM Puts for K < F and liquid OTM Calls for K >= F.
    4. Converts OTM Puts into synthetic European Calls via Put-Call Parity:
           C_equiv(K) = P_mid(K) + exp(-r * T) * (F - K)
    5. Filters static monotonicity violations (-exp(-r*T) < dC/dK < 0) and counts
       raw discrete butterfly spread violations (d^2C / dK^2 < 0).
    """
    calls_df, puts_df, spot, exp_str, T, source = fetch_raw_option_chain(
        ticker=ticker, target_dte=target_dte, use_live=use_live
    )

    raw_calls_n = len(calls_df)
    raw_puts_n = len(puts_df)

    forward, implied_q = estimate_implied_forward(calls_df, puts_df, spot, risk_free_rate, T)
    disc = np.exp(-risk_free_rate * T)

    # Prepare OTM Puts (K < F)
    puts = puts_df[puts_df["strike"] < forward].copy()
    puts["mid_price"] = 0.5 * (puts["bid"] + puts["ask"])
    puts["spread"] = puts["ask"] - puts["bid"]
    puts["rel_spread"] = puts["spread"] / np.maximum(puts["mid_price"], 1e-6)
    puts["option_type_used"] = "OTM_PUT"
    # Synthetic Call via Put-Call Parity: C(K) = P(K) + exp(-rT)*(F - K)
    puts["call_equiv_price"] = puts["mid_price"] + disc * (forward - puts["strike"])
    # Also record raw ITM call mid price (if available) to show why raw calls are inferior
    raw_c_lookup = calls_df.set_index("strike")[["bid", "ask"]].mean(axis=1)
    puts["raw_call_mid"] = puts["strike"].map(raw_c_lookup)

    # Prepare OTM Calls (K >= F)
    calls = calls_df[calls_df["strike"] >= forward].copy()
    calls["mid_price"] = 0.5 * (calls["bid"] + calls["ask"])
    calls["spread"] = calls["ask"] - calls["bid"]
    calls["rel_spread"] = calls["spread"] / np.maximum(calls["mid_price"], 1e-6)
    calls["option_type_used"] = "OTM_CALL"
    calls["call_equiv_price"] = calls["mid_price"]
    calls["raw_call_mid"] = calls["mid_price"]

    cols = [
        "strike",
        "option_type_used",
        "bid",
        "ask",
        "mid_price",
        "spread",
        "rel_spread",
        "call_equiv_price",
        "raw_call_mid",
    ]
    combined = pd.concat([puts[cols], calls[cols]], ignore_index=True).sort_values("strike").reset_index(drop=True)

    # 1. Liquidity & Moneyness Filter
    k_min, k_max = spot * moneyness_bounds[0], spot * moneyness_bounds[1]
    liquid = combined[
        (combined["bid"] > 0.01)
        & (combined["ask"] > combined["bid"])
        & (combined["rel_spread"] <= max_rel_spread)
        & (combined["strike"] >= k_min)
        & (combined["strike"] <= k_max)
    ].copy().reset_index(drop=True)

    # 2. Intrinsic Value Bound Filter: C(K) > max(0, exp(-rT)*(F - K))
    intrinsic = np.maximum(0.0, disc * (forward - liquid["strike"]))
    liquid["intrinsic_call"] = intrinsic
    liquid["time_value"] = liquid["call_equiv_price"] - intrinsic
    liquid = liquid[liquid["time_value"] > 0.005].copy().reset_index(drop=True)

    # If live market quotes are after-hours/stale and leave too few valid strikes,
    # fall back cleanly to the calibrated snapshot so the pipeline never crashes
    if len(liquid) < 30 and use_live:
        return clean_and_blend_otm_chain(
            ticker=ticker,
            target_dte=target_dte,
            risk_free_rate=risk_free_rate,
            max_rel_spread=max_rel_spread,
            moneyness_bounds=moneyness_bounds,
            use_live=False,
            save_csv_path=save_csv_path,
        )

    # 3. Enforce Strict Call Price Monotonicity: -exp(-rT) < dC/dK < 0
    keep_rows = [0]
    dropped_monotonicity = 0
    for i in range(1, len(liquid)):
        prev_idx = keep_rows[-1]
        dK = liquid.loc[i, "strike"] - liquid.loc[prev_idx, "strike"]
        dC = liquid.loc[i, "call_equiv_price"] - liquid.loc[prev_idx, "call_equiv_price"]
        slope = dC / dK
        if -disc < slope < -1e-6:
            keep_rows.append(i)
        else:
            dropped_monotonicity += 1

    cleaned = liquid.iloc[keep_rows].copy().reset_index(drop=True)
    if len(cleaned) < 25 and use_live:
        return clean_and_blend_otm_chain(
            ticker=ticker,
            target_dte=target_dte,
            risk_free_rate=risk_free_rate,
            max_rel_spread=max_rel_spread,
            moneyness_bounds=moneyness_bounds,
            use_live=False,
            save_csv_path=save_csv_path,
        )
    cleaned["moneyness"] = cleaned["strike"] / forward
    cleaned["log_moneyness"] = np.log(cleaned["moneyness"])

    # 4. Diagnostic: Compute Raw Discrete 1st & 2nd Derivatives (to illustrate the noise problem!)
    K_arr = cleaned["strike"].to_numpy()
    C_arr = cleaned["call_equiv_price"].to_numpy()
    dC_dK = np.gradient(C_arr, K_arr)

    # Non-uniform grid 2nd derivative finite difference
    d2C_dK2 = np.zeros_like(C_arr)
    for i in range(1, len(K_arr) - 1):
        h1 = K_arr[i] - K_arr[i - 1]
        h2 = K_arr[i + 1] - K_arr[i]
        d2C_dK2[i] = 2.0 * (
            (C_arr[i + 1] - C_arr[i]) / h2 - (C_arr[i] - C_arr[i - 1]) / h1
        ) / (h1 + h2)

    cleaned["raw_dC_dK"] = dC_dK
    cleaned["raw_d2C_dK2"] = d2C_dK2
    cleaned["raw_naive_rnd"] = np.exp(risk_free_rate * T) * d2C_dK2

    # Count interior strikes where naive finite difference gives negative density (butterfly violation)
    interior_rnd = cleaned["raw_naive_rnd"].iloc[1:-1]
    raw_butterfly_violations = int((interior_rnd < 0.0).sum())

    if save_csv_path is not None:
        out_path = Path(save_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(out_path, index=False)

    return CleanedOptionChain(
        ticker=ticker,
        expiration=exp_str,
        spot=spot,
        forward=forward,
        risk_free_rate=risk_free_rate,
        implied_dividend_yield=implied_q,
        time_to_expiry=T,
        df=cleaned,
        raw_calls_count=raw_calls_n,
        raw_puts_count=raw_puts_n,
        monotonicity_dropped=dropped_monotonicity,
        raw_butterfly_violations=raw_butterfly_violations,
        data_source=source,
    )


if __name__ == "__main__":
    out_csv = Path(__file__).resolve().parent.parent / "data" / "cleaned_spy_chain.csv"
    result = clean_and_blend_otm_chain(ticker="SPY", target_dte=45, save_csv_path=out_csv)

    print("=" * 72)
    print(f"PART 1 SUMMARY: OPTION CHAIN CLEANING & OTM BLENDING ({result.ticker})")
    print("=" * 72)
    print(f"Data Source           : {result.data_source} (Expiration: {result.expiration})")
    print(f"Spot Price (S0)       : ${result.spot:,.2f}")
    print(f"Implied Forward (F)   : ${result.forward:,.2f} (via Put-Call Parity)")
    print(f"Time to Expiry (T)    : {result.time_to_expiry * 365:.1f} days ({result.time_to_expiry:.4f} yrs)")
    print(f"Risk-Free Rate (r)    : {result.risk_free_rate:.2%}")
    print(f"Implied Div Yield (q) : {result.implied_dividend_yield:.2%}")
    print("-" * 72)
    print(f"Raw Quotes Pulled     : {result.raw_calls_count} Calls, {result.raw_puts_count} Puts")
    print(f"Cleaned OTM Strikes   : {len(result.df)} strikes (${result.df['strike'].min():.0f} to ${result.df['strike'].max():.0f})")
    print(f"Monotonicity Dropped  : {result.monotonicity_dropped} stale quotes removed")
    print(f"Raw Butterfly Traps   : {result.raw_butterfly_violations} strikes have d^2C/dK^2 < 0 (NEGATIVE PROBABILITY!)")
    print(f"Min / Max Naive RND   : [{result.df['raw_naive_rnd'].iloc[1:-1].min():.4f}, {result.df['raw_naive_rnd'].iloc[1:-1].max():.4f}]")
    print("-" * 72)
    print("Sample Cleaned Chain around ATM:")
    atm_view = result.df.iloc[(result.df["strike"] - result.forward).abs().argsort()[:8]].sort_values("strike")
    print(
        atm_view[
            ["strike", "option_type_used", "bid", "ask", "mid_price", "call_equiv_price", "raw_naive_rnd"]
        ].to_string(index=False)
    )
    print("=" * 72)
    print(f"Saved cleaned dataset to: {out_csv}")
