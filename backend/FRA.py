import numpy as np
import pandas as pd


def FRA(notional, alfa, payment_float, payment_fixed, discount_rate):
    PV = notional * alfa * (payment_float-payment_fixed) * discount_rate
    return PV

def FRA_settlement(notional, alfa, ref_rate, rate):
    FRA_sett = notional * alfa * (ref_rate - rate) / (1 + alfa * ref_rate)
    return FRA_sett

def forward_rates(df1, df2, alfa):
    return (df1 / df2 - 1) / alfa

def discount_factors(spot_rates, times):
    spot_rates = np.array(spot_rates)
    times = np.array(times)
    return np.exp(-spot_rates * times)

def DV01(notional, fixed_rate, r1, r2, t1, t2):
    alfa = t2 - t1
    bump = 0.0001

    df1 = discount_factors(r1, t1)
    df2 = discount_factors(r2, t2)
    forward_rate = forward_rates(df1, df2, alfa)

    PV1 = FRA(notional, alfa, forward_rate, fixed_rate, df2)
    df1_new = discount_factors(r1 + bump, t1)
    df2_new = discount_factors(r2 + bump, t2)

    forward_new = forward_rates(df1_new, df2_new, alfa)
    PV2 = FRA(notional, alfa, forward_new, fixed_rate, df2_new)

    return PV2 - PV1

def key_rate_dv01(notional, fixed_rate, r1, r2, t1, t2):
    alfa = t2 - t1
    bump = 0.0001

    def value(a, b):
        df1 = discount_factors(a, t1)
        df2 = discount_factors(b, t2)
        forward = forward_rates(df1, df2, alfa)
        return FRA(notional, alfa, forward, fixed_rate, df2)

    base = value(r1, r2)
    return pd.DataFrame({
        "Tenor": [f"{t1:g}Y", f"{t2:g}Y"],
        "Key Rate DV01": [value(r1 + bump, r2) - base,
                          value(r1, r2 + bump) - base]
    })
