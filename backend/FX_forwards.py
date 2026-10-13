import numpy as np


def fx_forwards(Spot, T, rate_curr1, rate_curr2):
    df1 = np.exp(-rate_curr1*T)
    df2 = np.exp(-rate_curr2*T)
    return Spot * (df1/df2)

def forward_curve(S, T, rate_curr1: list, rate_curr2: list):
    return fx_forwards(
        S,
        np.array(T),
        np.array(rate_curr1),
        np.array(rate_curr2)
    )

