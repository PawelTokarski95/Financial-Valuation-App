import numpy as np
import pandas as pd


def survival_probability(hazard_rate, T):
    return np.exp(-hazard_rate * T)


def default_probability(hazard_rate, T):
    return 1 - survival_probability(hazard_rate, T)


def cds_annuity(rate, hazard_rate, T):
    x = rate + hazard_rate

    if np.isclose(x, 0):
        return T

    return -np.expm1(-x * T) / x


def protection_leg(notional, recovery, hazard_rate, rate, T):
    annuity = cds_annuity(rate, hazard_rate, T)
    return notional * (1 - recovery) * hazard_rate * annuity


def premium_leg(notional, spread, hazard_rate, rate, T):
    annuity = cds_annuity(rate, hazard_rate, T)
    return notional * spread * annuity


def cds_npv(notional, spread, recovery, hazard_rate, rate, T):
    protection = protection_leg(notional, recovery, hazard_rate, rate, T)
    premium = premium_leg(notional, spread, hazard_rate, rate, T)

    return protection - premium


def cds_par_spread(hazard_rate, recovery):
    return hazard_rate * (1 - recovery)


def cds_calibrate(market_spread, recovery):
    if not 0 <= recovery < 1:
        raise ValueError("Recovery must be between 0 and 1.")

    return market_spread / (1 - recovery)


def cds_cs01(notional, spread, recovery, hazard_rate, rate, T):
    bump = 0.0001

    base = cds_npv(notional, spread, recovery, hazard_rate, rate, T)

    market_spread = cds_par_spread(hazard_rate, recovery)
    new_hazard = cds_calibrate(market_spread + bump, recovery)

    shocked = cds_npv(notional, spread, recovery, new_hazard, rate, T)

    return shocked - base


def cds_ir_dv01(notional, spread, recovery, hazard_rate, rate, T):
    bump = 0.0001

    base = cds_npv(notional, spread, recovery, hazard_rate, rate, T)
    shocked = cds_npv(notional, spread, recovery, hazard_rate, rate + bump, T)

    return shocked - base


def cds_scenarios(notional, spread, rate, T, hazards, recoveries):
    results = []

    for hazard in hazards:
        for recovery in recoveries:
            npv = cds_npv(notional, spread, recovery, hazard, rate, T)
            pd_value = default_probability(hazard, T)

            results.append([hazard, recovery, pd_value, npv])

    return pd.DataFrame(
        results,
        columns=["hazard", "recovery", "PD", "NPV"]
    )


def cds_report(notional, spread, recovery, hazard_rate, rate, T):
    return {
        "survival_probability": survival_probability(hazard_rate, T),
        "default_probability": default_probability(hazard_rate, T),
        "protection_leg": protection_leg(notional, recovery, hazard_rate, rate, T),
        "premium_leg": premium_leg(notional, spread, hazard_rate, rate, T),
        "npv": cds_npv(notional, spread, recovery, hazard_rate, rate, T),
        "par_spread": cds_par_spread(hazard_rate, recovery),
        "cs01": cds_cs01(notional, spread, recovery, hazard_rate, rate, T),
        "ir_dv01": cds_ir_dv01(notional, spread, recovery, hazard_rate, rate, T)
    }
