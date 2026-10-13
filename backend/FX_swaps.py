
import numpy as np
import pandas as pd


def discount_factor(rate, T):
    return np.exp(-rate * T)


def swap_points(spot, forward):
    return forward - spot


def df(rate, T):
    return np.exp(-rate * T)


def fx_forward(spot, r_eur, r_pln, T):
    return spot * df(r_eur, T) / df(r_pln, T)


def fx_swap_npv(nominal, spot, near_rate, far_rate,
                t_near, t_far, r_eur, r_pln):

    near = nominal * (
        spot * df(r_eur, t_near)
        - near_rate * df(r_pln, t_near)
    )

    far = nominal * (
        far_rate * df(r_pln, t_far)
        - spot * df(r_eur, t_far)
    )

    return near + far


def fx_delta(nominal, t_near, t_far, r_eur):
    return nominal * (
        df(r_eur, t_near) - df(r_eur, t_far)
    )


def fx_swap_dv01(nominal, spot, near_rate, far_rate,
                 t_near, t_far, r_eur, r_pln):

    base = fx_swap_npv(
        nominal, spot, near_rate, far_rate,
        t_near, t_far, r_eur, r_pln
    )

    eur_shock = fx_swap_npv(
        nominal, spot, near_rate, far_rate,
        t_near, t_far, r_eur + 0.0001, r_pln
    )

    pln_shock = fx_swap_npv(
        nominal, spot, near_rate, far_rate,
        t_near, t_far, r_eur, r_pln + 0.0001
    )

    return {
        "EUR_DV01": eur_shock - base,
        "PLN_DV01": pln_shock - base
    }


def forward_points(spot, r_eur, r_pln, T):
    return fx_forward(spot, r_eur, r_pln, T) - spot


def calibrate_fx(spot, market_points, r_pln, T):
    market_forward = spot + market_points
    df_eur = market_forward / spot * df(r_pln, T)

    return -np.log(df_eur) / T


def fx_scenarios(nominal, spot, near_rate, far_rate,
                 t_near, t_far, r_eur, r_pln, shocks):

    results = []

    for shock in shocks:
        new_spot = spot * (1 + shock)

        npv = fx_swap_npv(
            nominal, new_spot, near_rate, far_rate,
            t_near, t_far, r_eur, r_pln
        )

        results.append([shock, new_spot, npv])

    return pd.DataFrame(
        results, columns=["shock", "spot", "npv"]
    )
