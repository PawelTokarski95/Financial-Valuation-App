try:
    import QuantLib as ql
except ImportError:
    ql = None
import math
import datetime as dt

import numpy as np
import pandas as pd
from scipy import optimize
import plotly.graph_objects as go

def heston(
    valuation_date: dt.datetime,
    maturity_date: dt.datetime,
    r: float,
    q: float,
    S0: float,
    K: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call"
):

    if ql is None:
        raise ImportError("Model Hestona wymaga pakietu QuantLib: pip install QuantLib")

    # 1. Walidacja danych
    if maturity_date <= valuation_date:
        raise ValueError("Maturity date must be after valuation date.")

    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")

    if v0 < 0 or theta < 0 or kappa <= 0 or xi <= 0:
        raise ValueError("Invalid Heston variance parameters.")

    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1.")

    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # 2. Daty QuantLib
    valuation_date = ql.Date(
        valuation_date.day,
        valuation_date.month,
        valuation_date.year
    )

    maturity_date = ql.Date(
        maturity_date.day,
        maturity_date.month,
        maturity_date.year
    )

    ql.Settings.instance().evaluationDate = valuation_date

    # 3. Dane rynkowe
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    day_count = ql.Actual365Fixed()

    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(valuation_date, r, day_count)
    )

    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(valuation_date, q, day_count)
    )

    # 4. Proces stochastyczny Hestona
    heston_process = ql.HestonProcess(
        risk_free_ts,
        dividend_ts,
        spot,
        v0,
        kappa,
        theta,
        xi,
        rho
    )

    model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(model)

    payoff_type = (
        ql.Option.Call
        if option_type == "call"
        else ql.Option.Put
    )

    payoff = ql.PlainVanillaPayoff(payoff_type, K)
    exercise = ql.EuropeanExercise(maturity_date)

    option = ql.VanillaOption(payoff, exercise)

    option.setPricingEngine(engine)

    return float(option.NPV())




def black_scholes(
    valuation_date: dt.datetime,
    maturity_date: dt.datetime,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call"
):

    if maturity_date <= valuation_date:
        raise ValueError("Maturity must be after valuation date.")

    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")

    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    option_type = option_type.lower()

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    T = (maturity_date - valuation_date).total_seconds() / (365 * 24 * 3600)

    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    d1 = (
        math.log(S0 / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    discounted_spot = S0 * math.exp(-q * T)
    discounted_strike = K * math.exp(-r * T)

    if option_type == "call":
        price = (
            discounted_spot * norm_cdf(d1)
            - discounted_strike * norm_cdf(d2)
        )

    else:
        price = (
            discounted_strike * norm_cdf(-d2)
            - discounted_spot * norm_cdf(-d1)
        )

    return float(price)

def crr(
    valuation_date: dt.datetime,
    maturity_date: dt.datetime,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    steps: int = 200,
    option_type: str = "call",
    exercise: str = "european"
):

    if maturity_date <= valuation_date:
        raise ValueError("Maturity must be after valuation date.")

    if S0 <= 0 or K <= 0 or sigma <= 0:
        raise ValueError("S0, K and sigma must be positive.")

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("Steps must be a positive integer.")

    option_type = option_type.lower()
    exercise = exercise.lower()

    if option_type not in ("call", "put"):
        raise ValueError("Option must be 'call' or 'put'.")

    if exercise not in ("european", "american"):
        raise ValueError("Exercise must be 'european' or 'american'.")

    T = (maturity_date - valuation_date).total_seconds() / (365 * 24 * 3600)

    dt_step = T / steps

    u = math.exp(sigma * math.sqrt(dt_step))
    d = 1 / u

    p = (math.exp((r - q) * dt_step) - d) / (u - d)

    if not 0 <= p <= 1:
        raise ValueError("Invalid risk-neutral probability. Increase steps.")

    discount = math.exp(-r * dt_step)

    sign = 1 if option_type == "call" else -1

    values = []

    for j in range(steps + 1):
        S_T = S0 * (u ** j) * (d ** (steps - j))
        payoff = max(sign * (S_T - K), 0.0)
        values.append(payoff)

    for i in range(steps - 1, -1, -1):

        for j in range(i + 1):

            continuation = discount * (
                p * values[j + 1] + (1 - p) * values[j]
            )

            if exercise == "american":
                S = S0 * (u ** j) * (d ** (i - j))
                immediate_exercise = max(sign * (S - K), 0.0)

                values[j] = max(
                    continuation,
                    immediate_exercise
                )

            else:
                values[j] = continuation

    return float(values[0])



def implied_volatility(
    valuation_date: dt.datetime,
    maturity_date: dt.datetime,
    r: float,
    q: float,
    S0: float,
    K: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call"):

    price = heston(
        valuation_date=valuation_date,
        maturity_date=maturity_date,
        r=r,
        q=q,
        S0=S0,
        K=K,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        option_type=option_type
    )

    def objective(sigma):
        return black_scholes(
            valuation_date=valuation_date,
            maturity_date=maturity_date,
            S0=S0,
            K=K,
            r=r,
            q=q,
            sigma=sigma,
            option_type=option_type) - price

    IV = optimize.brentq(objective, 0.0001, 4.0)

    return float(IV)


def volatility_smile(
    valuation_date: dt.datetime,
    maturity_date: dt.datetime,
    S0: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_strikes: int = 50,
    option_type: str = "call"
):
    strikes = np.linspace(S0 * 0.7, S0 * 1.3, n_strikes)

    results = []

    for K in strikes:
        iv = implied_volatility(
            valuation_date=valuation_date,
            maturity_date=maturity_date,
            r=r,
            q=q,
            S0=S0,
            K=float(K),
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            option_type=option_type
        )

        results.append({
            "Strike": K,
            "IV": iv
        })

    df = pd.DataFrame(results)

    # 3. Tworzenie wykresu
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Strike"],
        y=df["IV"],
        mode="lines+markers",
        name="Heston Implied Volatility",
        line=dict(
            color="#2563EB",
            width=3
        ),
        marker=dict(
            color="#1D4ED8",
            size=5,
            line=dict(color="white", width=1)
        ),
        hovertemplate=(
            "<b>Strike:</b> %{x:.2f} PLN"
            "<br><b>Implied Volatility:</b> %{y:.2%}"
            "<extra></extra>"
        )
    ))

    # 4. Linia ATM
    fig.add_vline(
        x=S0,
        line_width=1.5,
        line_dash="dash",
        line_color="#94A3B8",
        annotation_text="ATM (Spot)",
        annotation_position="top right"
    )

    # 5. Referencyjna zmienność początkowa
    fig.add_hline(
        y=np.sqrt(v0),
        line_dash="dot",
        line_width=1.2,
        line_color="#64748B",
        annotation_text="Initial Volatility",
        annotation_position="bottom right"
    )

    # 6. Profesjonalny wygląd
    fig.update_layout(
        title=dict(
            text="<b>Heston Volatility Smile</b>",
            x=0.05,
            font=dict(size=23, color="#0F172A")
        ),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            size=13,
            color="#334155"
        ),
        height=550,
        margin=dict(l=70, r=40, t=90, b=70),
        hovermode="closest",
        showlegend=False
    )

    fig.update_xaxes(
        title="Strike Price (PLN)",
        showgrid=True,
        gridcolor="#E2E8F0",
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1"
    )

    fig.update_yaxes(
        title="Implied Volatility",
        tickformat=".1%",
        showgrid=True,
        gridcolor="#E2E8F0",
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1"
    )


    return df, fig

