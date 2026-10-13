import numpy as np
import pandas as pd


def discount_factors(spot_rates, times):
    spot_rates = np.array(spot_rates)
    times = np.array(times)
    return np.exp(-spot_rates * times)


def forward_rates(df, alfa):
    df = np.array(df)
    previous = np.insert(df[:-1], 0, 1)
    return (previous / df - 1) / alfa


def swap(nominal, rate, alfa, df, forwards, side="payer"):
    fixed = nominal * rate * np.sum(alfa * df)
    floating = nominal * np.sum(alfa * forwards * df)

    if side == "payer":
        npv = floating - fixed
    else:
        npv = fixed - floating

    return {
        "fixed": fixed,
        "floating": floating,
        "npv": npv
    }


def par_swap_rate(alfa, df, forwards):
    return np.sum(alfa * forwards * df) / np.sum(alfa * df)


def payment_schedule(maturity, frequency=2):
    n = int(maturity * frequency)
    alfa = np.full(n, 1 / frequency)
    times = np.arange(1, n + 1) / frequency
    return times, alfa


def DV01(nominal, rate, spot_rates, times, alfa, side="payer"):
    df = discount_factors(spot_rates, times)
    forwards = forward_rates(df, alfa)
    base = swap(nominal, rate, alfa, df, forwards, side)["npv"]

    new_rates = np.array(spot_rates) + 0.0001
    new_df = discount_factors(new_rates, times)
    new_forwards = forward_rates(new_df, alfa)
    shocked = swap(nominal, rate, alfa, new_df, new_forwards, side)["npv"]

    return shocked - base


def key_rate_dv01(nominal, rate, spot_rates, times, alfa, side="payer"):
    spot_rates = np.array(spot_rates)
    df = discount_factors(spot_rates, times)
    forwards = forward_rates(df, alfa)
    base = swap(nominal, rate, alfa, df, forwards, side)["npv"]

    results = []

    for i in range(len(spot_rates)):
        new_rates = spot_rates.copy()
        new_rates[i] += 0.0001

        new_df = discount_factors(new_rates, times)
        new_forwards = forward_rates(new_df, alfa)
        shocked = swap(nominal, rate, alfa, new_df, new_forwards, side)["npv"]

        results.append([times[i], shocked - base])

    return pd.DataFrame(results, columns=["tenor", "dv01"])


def scenario_analysis(nominal, rate, spot_rates, times, alfa, shocks, side="payer"):
    results = []

    for shock in shocks:
        new_rates = np.array(spot_rates) + shock / 10000
        df = discount_factors(new_rates, times)
        forwards = forward_rates(df, alfa)

        npv = swap(nominal, rate, alfa, df, forwards, side)["npv"]
        par = par_swap_rate(alfa, df, forwards)

        results.append([shock, npv, par])

    return pd.DataFrame(results, columns=["shock", "npv", "par_rate"])


