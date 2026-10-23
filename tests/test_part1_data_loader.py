"""
Unit tests for Part 1: Option Chain Data Pipeline, Implied Forward & OTM Blending.
"""

import numpy as np
from src.data_loader import clean_and_blend_otm_chain


def test_clean_and_blend_otm_chain_snapshot():
    """Verifies that Part 1 produces a clean, strictly decreasing call price curve across OTM strikes."""
    chain = clean_and_blend_otm_chain(ticker="SPY", target_dte=45, use_live=False)

    assert chain.spot > 0
    assert chain.forward > 0
    assert len(chain.df) >= 40

    # Verify OTM Puts are used below Forward and OTM Calls above Forward
    below_f = chain.df[chain.df["strike"] < chain.forward]
    above_f = chain.df[chain.df["strike"] >= chain.forward]
    assert (below_f["option_type_used"] == "OTM_PUT").all()
    assert (above_f["option_type_used"] == "OTM_CALL").all()

    # Verify strict call price monotonicity: -exp(-rT) < dC/dK < 0
    disc = np.exp(-chain.risk_free_rate * chain.time_to_expiry)
    dK = np.diff(chain.df["strike"].to_numpy())
    dC = np.diff(chain.df["call_equiv_price"].to_numpy())
    slopes = dC / dK
    assert np.all(slopes < 0.0), "Call equivalent prices must strictly decrease with strike K"
    assert np.all(slopes > -disc), "Call slope cannot violate -exp(-rT) lower bound"

    # Verify that raw discrete 2nd derivative exhibits butterfly noise (motivating Part 2 IV smoothing)
    assert chain.raw_butterfly_violations > 0
