import plotly.express as px
from scipy.optimize import brentq


def spot_curve(spot_rates: list):
    periods = list(range(1, len(spot_rates) + 1))

    fig = px.line(
        x=periods,
        y=spot_rates,
        markers=True,
        title="Spot Rate Curve",
        labels={"x": "Maturity (years)", "y": "Spot Rate"},
        template="plotly_white"
    )

    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(hovermode="x unified")


def forward_curve(spot_rates: list):
    forward_rates = [spot_rates[0]]

    for period in range(1, len(spot_rates)):
        t = period + 1

        forward = (
            (1 + spot_rates[period]) ** t
            / (1 + spot_rates[period - 1]) ** (t - 1)
            - 1
        )

        forward_rates.append(forward)

    fig = px.line(
        x=list(range(1, len(forward_rates) + 1)),
        y=forward_rates,
        markers=True,
        title="Forward Rate Curve",
        labels={"x": "Forward year", "y": "Forward Rate"},
        template="plotly_white"
    )

    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(hovermode="x unified")

    return forward_rates


def bond_price(ytm, coupon_rate, maturity, notional=1000):
    coupon = coupon_rate * notional

    pv = sum(
        coupon / (1 + ytm) ** time
        for time in range(1, maturity + 1)
    )

    pv_notional = notional / (1 + ytm) ** maturity

    return pv + pv_notional


def ytm(price, coupon_rate, maturity, nominal=1000):
    return brentq(
        lambda x: bond_price(
            x, coupon_rate, maturity, nominal
        ) - price,
        -0.9,
        1.0
    )


def yield_curve(bonds):
    bonds = bonds.copy()

    bonds["ytm"] = bonds.apply(
        lambda row: ytm(
            row["price"],
            row["coupon_rate"],
            int(row["maturity"])
        ),
        axis=1
    )

    bonds = bonds.sort_values("maturity")

    fig = px.line(
        bonds,
        x="maturity",
        y="ytm",
        markers=True,
        title="Bond Yield Curve",
        labels={
            "maturity": "Maturity (years)",
            "ytm": "Yield to Maturity"
        },
        template="plotly_white"
    )

    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(hovermode="x unified")

    return bonds

