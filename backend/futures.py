import numpy as np
import pandas as pd


def futures_price(spot, maturity, rate, income_yield=0.0,
                  storage_rate=0.0, convenience_yield=0.0):
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("Spot price must be positive and finite.")
    if not np.isfinite(maturity) or maturity < 0:
        raise ValueError("Maturity must be non-negative and finite.")
    rates = (rate, income_yield, storage_rate, convenience_yield)
    if not all(np.isfinite(v) for v in rates):
        raise ValueError("Rates must be finite.")
    carry = rate + storage_rate - income_yield - convenience_yield
    return float(spot * np.exp(carry * maturity))


def fx_futures_price(spot, maturity, quote_rate, base_rate):
    return futures_price(spot, maturity, quote_rate, income_yield=base_rate)


def futures_pnl(market_price, entry_price, contracts, multiplier=1.0,
                side="long"):
    if side not in ("long", "short"):
        raise ValueError("Position must be long or short.")
    if contracts <= 0 or multiplier <= 0:
        raise ValueError("Contracts and multiplier must be positive.")
    if market_price <= 0 or entry_price <= 0:
        raise ValueError("Market and entry prices must be positive.")
    direction = 1 if side == "long" else -1
    return float(direction * contracts * multiplier * (market_price - entry_price))


def futures_delta(spot, maturity, rate, contracts, multiplier,
                  income_yield=0.0, storage_rate=0.0,
                  convenience_yield=0.0, side="long"):
    direction = 1 if side == "long" else -1
    fair = futures_price(spot, maturity, rate, income_yield,
                         storage_rate, convenience_yield)
    return float(direction * contracts * multiplier * fair / spot)


def futures_rate_dv01(spot, maturity, rate, contracts, multiplier,
                      income_yield=0.0, storage_rate=0.0,
                      convenience_yield=0.0, side="long", bump=0.0001):
    fair = futures_price(spot, maturity, rate, income_yield,
                         storage_rate, convenience_yield)
    shocked = futures_price(spot, maturity, rate + bump, income_yield,
                            storage_rate, convenience_yield)
    direction = 1 if side == "long" else -1
    return float(direction * contracts * multiplier * (shocked - fair))


def futures_scenarios(spot, maturity, rate, entry_price, contracts,
                      multiplier=1.0, side="long", income_yield=0.0,
                      storage_rate=0.0, convenience_yield=0.0,
                      spot_changes=(-0.10, -0.05, 0.0, 0.05, 0.10),
                      rate_changes_bps=(-100, 0, 100)):
    results = []
    base_fair = futures_price(spot, maturity, rate, income_yield,
                              storage_rate, convenience_yield)
    direction = 1 if side == "long" else -1
    for s in spot_changes:
        for bps in rate_changes_bps:
            fair = futures_price(spot * (1 + s), maturity, rate + bps / 10000,
                                 income_yield, storage_rate, convenience_yield)
            results.append({
                "Spot shock (%)": 100 * s,
                "Rate shock (bp)": bps,
                "Model futures price": fair,
                "Price change vs base": fair - base_fair,
                "Illustrative P&L vs entry": direction * contracts * multiplier * (fair - entry_price)
            })
    return pd.DataFrame(results)
