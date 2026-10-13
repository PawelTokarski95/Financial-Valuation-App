from datetime import date, datetime, time, timedelta
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from backend import bonds, CDS, FRA, FX_forwards, FX_swaps, futures, options, stocks, swaps

st.set_page_config(page_title="Financial Valuation App", layout="wide")

BLUE = "#2563EB"
DARK = "#0F172A"
RED = "#DC2626"
GREEN = "#059669"

st.markdown("""
<style>
.block-container {max-width: 1640px; padding: 1.3rem 2.1rem 3.5rem;}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse at top left, #EEF3FF 0, #F6F8FC 470px, #F6F8FC 100%);}
[data-testid="stHeader"] {background: transparent;}
h1, h2, h3 {color: #182945; letter-spacing: -0.025em;}
h3 {font-size: 1.15rem;}
.hero {background: linear-gradient(110deg, #10274E 0%, #1A3A73 68%, #2451A1 100%);
  padding: 26px 30px; border-radius: 18px; margin-bottom: 20px; color: white;
  box-shadow: 0 10px 28px rgba(13, 32, 73, .13);}
.hero-kicker {font-size: .72rem; letter-spacing: .18em; text-transform: uppercase;
  color: #C5D8FF; font-weight: 700; margin-bottom: 8px;}
.hero-title {font-size: 2rem; font-weight: 780; line-height: 1.15; letter-spacing: -.035em; margin-bottom: 8px;}
.hero-subtitle {color: #D9E6FF; font-size: .95rem; line-height: 1.55;}
[data-testid="stMetric"] {background: white; border: 1px solid #E1E8F3;
  box-shadow: 0 2px 10px rgba(31, 58, 94, .035);
  border-radius: 13px; padding: 14px 17px; min-height: 96px;}
[data-testid="stMetricLabel"] {color: #657590; font-size: .84rem;}
[data-testid="stMetricValue"] {font-size: clamp(1.25rem, 2vw, 1.8rem); color: #17335F;}
[data-testid="stExpander"] {background: #FFFFFF; border: 1px solid #E1E8F3;
  border-radius: 13px; box-shadow: 0 3px 12px rgba(25, 46, 81, .035); overflow: hidden;}
[data-testid="stExpander"] details summary {padding-top: .30rem; padding-bottom: .30rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color: #DFE7F2; border-radius: 15px;}

[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"],
[data-testid="stDateInput"] [data-baseweb="datepicker"],
[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stNumberInput"] [data-baseweb="base-input"] {
  border-radius: 10px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within,
[data-testid="stDateInput"] [data-baseweb="datepicker"]:focus-within,
[data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within,
[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within {
  box-shadow: 0 0 0 3px rgba(37, 99, 235, .10);
}

[data-testid="stMultiSelect"] [data-baseweb="tag"] {background: #DDE9FF;
  color: #194B9B; border-radius: 7px;}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {color: #194B9B;}
[data-testid="stPlotlyChart"] {background: #FFFFFF; border: 1px solid #E4EAF3;
  border-radius: 12px; padding: 7px;}
[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
[data-testid="stDownloadButton"] button {border-radius: 9px; border-color: #C9D7ED; color: #244F98;}
[data-testid="stAlert"] {border-radius: 11px;}
@media (max-width: 800px) {
  .block-container {padding: .9rem .8rem 2rem;}
  .hero {padding: 22px 18px; border-radius: 14px;}
  .hero-title {font-size: 1.55rem;}
}
</style>
""", unsafe_allow_html=True)


def dot_number_input(label, *, value, min_value=None, max_value=None,
                     step=None, format=None, key=None, **kwargs):
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        options = {"value": int(value), "key": key, **kwargs}
        if min_value is not None:
            options["min_value"] = min_value
        if max_value is not None:
            options["max_value"] = max_value
        if step is not None:
            options["step"] = step
        return st.number_input(label, **options)

    places = 2
    if format:
        match = re.search(r"\.(\d+)f", format)
        if match:
            places = int(match.group(1))
    elif value != 0:
        normalized = f"{float(value):.10f}".rstrip("0").rstrip(".")
        places = max(2, min(8, len(normalized.split(".")[1]) if "." in normalized else 2))

    default = f"{float(value):.{places}f}"
    raw = st.text_input(label, value=default, key=key,
                        help="Use a period (.) as the decimal separator.")

    try:
        if not isinstance(raw, (int, float)) and "," in str(raw):
            raise ValueError("Use a period (.) for decimals, not a comma.")
        result = float(str(raw).strip())
        if not np.isfinite(result):
            raise ValueError("The value must be finite.")
        if min_value is not None and result < min_value:
            raise ValueError(f"Enter a value of at least {min_value}.")
        if max_value is not None and result > max_value:
            raise ValueError(f"Enter a value no greater than {max_value}.")
        return result
    except (ValueError, TypeError) as error:
        st.error(f"{label}: {error}")
        return float(value)


def percent(label, key, default, *, min_value=None):
    kw = {"value": float(default), "step": 0.25, "format": "%.2f", "key": key}
    if min_value is not None:
        kw["min_value"] = float(min_value)
    return dot_number_input(label, **kw) / 100.0


def bps(label, key, default):
    return dot_number_input(label, value=float(default), step=5.0, key=key) / 10000.0


def parse_percent_list(raw):
    if ";" in raw:
        parts = [part.strip().replace(",", ".") for part in raw.split(";")]
    else:
        parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("Enter rates separated by commas, e.g. 4.0, 4.2, 4.5")
    try:
        values = np.array([float(part) / 100 for part in parts], dtype=float)
    except ValueError as error:
        raise ValueError("Use commas between values and periods for decimals: 4.0, 4.2, 4.5") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("All rates must be finite numbers.")
    return values


def chart(fig, title=None, *, percent_y=False):
    fig.update_layout(
        template="plotly_white", height=365, margin=dict(l=20, r=20, t=54, b=25),
        font=dict(family="Arial", color=DARK),
        title=dict(text=title, font=dict(size=17)) if title else None,
        hovermode="x unified", legend=dict(orientation="h", y=-0.25),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5EAF1", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5EAF1", zeroline=False)
    if percent_y:
        fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)


def line_plot(x, y, name, *, title, xname, yname, percent_y=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name,
                             line=dict(color=BLUE, width=3), marker=dict(size=7)))
    fig.update_xaxes(title_text=xname)
    fig.update_yaxes(title_text=yname)
    chart(fig, title, percent_y=percent_y)


def metric(label, value, unit="", decimals=2):
    st.metric(label, f"{float(value):,.{decimals}f}".replace(",", " ") + unit)


def table_download(df, key, filename="results.csv"):
    display_df = df.rename(columns=lambda name: str(name).replace("&", "and"))
    formatted = display_df.copy()
    for name in formatted.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_integer_dtype(formatted[name]):
            formatted[name] = formatted[name].map(lambda value: str(value))
        else:
            formatted[name] = formatted[name].map(
                lambda value: f"{value:.6f}".rstrip("0").rstrip(".")
                if pd.notna(value) else ""
            )
    st.dataframe(formatted, use_container_width=True, hide_index=True)
    st.download_button("Export CSV", display_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv", key=key)


def show_inputs(inputs, _function_info=None):
    with st.expander("Valuation inputs"):
        st.json({k: (v.tolist() if isinstance(v, np.ndarray) else str(v) if isinstance(v, date) else v)
                 for k, v in inputs.items()})


INSTRUMENTS = ["stocks", "bonds", "options", "Futures", "CDS", "Swaps", "FRA",
               "FX Forward", "FX Futures", "FX Swaps"]
DISPLAY_NAMES = {
    "stocks": "Stocks", "bonds": "Bonds", "options": "Options",
    "Futures": "Futures", "CDS": "Credit Default Swaps", "Swaps": "Interest Rate Swaps",
    "FRA": "Forward Rate Agreements", "FX Forward": "FX Forward",
    "FX Futures": "FX Futures", "FX Swaps": "FX Swaps"
}


def futures_parameters(prefix, fx_only=False):
    p = {}
    p["kind"] = ("FX Futures" if fx_only else st.selectbox(
        "Contract category", ["Equity Index", "Commodity", "FX Futures"], key=f"{prefix}_kind"))
    is_fx = p["kind"] == "FX Futures"
    commodity = p["kind"] == "Commodity"
    p["spot"] = dot_number_input("Underlying spot price", min_value=0.00001,
                                value=4.30 if is_fx else 80.0 if commodity else 5000.0,
                                key=f"{prefix}_spot")
    p["market"] = dot_number_input("Current futures market quote", min_value=0.00001,
                                  value=4.38 if is_fx else 81.50 if commodity else 5100.0,
                                  key=f"{prefix}_market")
    p["entry"] = dot_number_input("Position entry price", min_value=0.00001,
                                 value=4.35 if is_fx else 80.50 if commodity else 5000.0,
                                 key=f"{prefix}_entry")
    p["T"] = dot_number_input("Time to expiration (years)", min_value=0.001,
                             value=1.0 if is_fx else 0.5, key=f"{prefix}_maturity")
    p["contracts"] = dot_number_input("Number of contracts", min_value=1, value=1 if is_fx else 2,
                                     step=1, key=f"{prefix}_contracts")
    p["multiplier"] = dot_number_input("Contract multiplier", min_value=0.01,
                                     value=100000.0 if is_fx else 1000.0 if commodity else 50.0,
                                     key=f"{prefix}_multiplier")
    p["side"] = st.selectbox("Position direction", ["long", "short"], key=f"{prefix}_side")
    p["rate"] = percent("Quote currency risk-free rate (%)" if is_fx else "Financing rate (%)",
                        f"{prefix}_rate", 5 if is_fx else 4.5)
    p["income"] = percent("Base currency risk-free rate (%)" if is_fx else
                          "Income / dividend yield (%)" if not commodity else "Income yield (%)",
                          f"{prefix}_income", 3 if is_fx else 0 if commodity else 1.5)
    p["storage"] = percent("Annual storage costs (%)", f"{prefix}_storage", 1.0) if commodity else 0.0
    p["convenience"] = percent("Convenience yield (%)", f"{prefix}_convenience", 2.0) if commodity else 0.0
    p["margin"] = dot_number_input("Initial margin per contract (optional)", min_value=0.0,
                                  value=5000.0 if is_fx else 6000.0 if commodity else 10000.0,
                                  key=f"{prefix}_margin")
    return p


def futures_results(p, instrument):
    rate = p["rate"]
    income = p["income"]
    storage = p["storage"]
    convenience = p["convenience"]
    fair = futures.futures_price(p["spot"], p["T"], rate, income, storage, convenience)
    trade_pnl = futures.futures_pnl(p["market"], p["entry"], p["contracts"],
                                    p["multiplier"], p["side"])
    delta = futures.futures_delta(p["spot"], p["T"], rate, p["contracts"],
                                  p["multiplier"], income, storage, convenience, p["side"])
    ir_dv01 = futures.futures_rate_dv01(p["spot"], p["T"], rate, p["contracts"],
                                        p["multiplier"], income, storage, convenience, p["side"])
    a, b, c = st.columns(3)
    with a: metric("Theoretical futures price", fair, decimals=5 if p["kind"] == "FX Futures" else 2)
    with b: metric("Market minus fair price", p["market"] - fair, decimals=5 if p["kind"] == "FX Futures" else 2)
    with c: metric("Price movement profit and loss vs entry", trade_pnl)
    a, b, c = st.columns(3)
    with a: metric("Underlying spot Delta", delta)
    with b: metric("Financing DV01 (+1 bp)", ir_dv01)
    with c: metric("Initial margin estimate", p["margin"] * p["contracts"])
    tenors = np.linspace(0, max(p["T"] * 1.7, 0.15), 55)
    curve = [futures.futures_price(p["spot"], t, rate, income, storage, convenience) for t in tenors]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tenors, y=curve, mode="lines", name="Model price",
                             line=dict(color=BLUE, width=3)))
    fig.add_hline(y=p["market"], line_dash="dot", line_color="#8196B1",
                  annotation_text="Market quote", annotation_position="top right")
    fig.update_xaxes(title_text="Time to expiration (years)")
    fig.update_yaxes(title_text="Futures price")
    chart(fig, "Futures term structure and current market quote")
    scenarios = futures.futures_scenarios(
        p["spot"], p["T"], rate, p["entry"], p["contracts"], p["multiplier"],
        p["side"], income, storage, convenience
    )
    base = scenarios[scenarios["Rate shock (bp)"] == 0]
    line_plot(base["Spot shock (%)"], base["Illustrative P&L vs entry"], "Scenario profit and loss",
              title="Scenario profit and loss vs underlying spot change", xname="Spot change (%)",
              yname="Illustrative profit and loss")
    pivot = scenarios.pivot(index="Spot shock (%)", columns="Rate shock (bp)",
                            values="Model futures price")
    fig = go.Figure(data=go.Heatmap(x=pivot.columns, y=pivot.index,
                                   z=pivot.to_numpy(), colorscale="Blues",
                                   colorbar=dict(title="Price")))
    chart(fig, "Futures model price — spot and financing shocks")
    table_download(scenarios.round(5), f"{instrument.replace(' ', '_')}_scenario_export", "futures_scenarios.csv")
    if p["kind"] == "FX Futures":
        st.caption("FX futures carry benchmark under deterministic interest rates. "
                   "Daily settlement and rate/FX correlation can create a futures-forward difference.")
    else:
        st.caption("Cost-of-carry benchmark. Daily variation margin and contract specifications "
                   "may affect actual results. Price movement profit and loss is not the value of an unsettled forward.")
    st.caption("Spot Delta measures profit and loss per one-unit spot move assuming repricing to the theoretical quote. "
               "DV01 measures a +1 bp financing-rate shock.")
    show_inputs(p)


st.markdown("""
<div class="hero">
  <div class="hero-title">Financial Valuation App</div>
</div>
""", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1.5], gap="large")
params = {}

with col1:
    st.subheader("Model Parameters")
    st.caption("Choose instruments and set your valuation assumptions.")
    selected = st.multiselect("Instruments", INSTRUMENTS,
                              default=["stocks"], format_func=lambda x: DISPLAY_NAMES[x],
                              key="instrument_selector")

    if "stocks" in selected:
        with st.expander("Stocks · Equity Valuation", expanded=True):
            p = {}
            p["model"] = st.selectbox("Valuation method", ["DCF (FCFF)", "DDM", "P/E"], key="stocks_model")
            p["market_price"] = dot_number_input("Current share price", min_value=0.01, value=65.0, key="stocks_market")
            if p["model"] == "DCF (FCFF)":
                p["fcff0"] = dot_number_input("Most recent annual FCFF", value=350_000_000.0, key="stocks_fcff")
                p["growth_raw"] = st.text_input("Forecast FCFF growth (%; comma-separated)",
                                                "8, 7, 6, 5, 4", key="stocks_growth")
                p["estimate_wacc"] = st.checkbox("Calculate WACC using CAPM", value=True, key="stocks_capm")
                if p["estimate_wacc"]:
                    p["rf"] = percent("Risk-free rate (%)", "stocks_rf", 4)
                    p["beta"] = dot_number_input("Beta", value=1.10, step=0.05, key="stocks_beta")
                    p["erp"] = percent("Equity risk premium (%)", "stocks_erp", 5.5)
                    p["kd"] = percent("Cost of debt (%)", "stocks_kd", 6.5)
                    p["tax"] = percent("Corporate tax rate (%)", "stocks_tax", 19, min_value=0)
                    p["we"] = percent("Equity weight (%)", "stocks_we", 75, min_value=0)
                else:
                    p["wacc"] = percent("WACC (%)", "stocks_wacc", 8.85)
                p["g"] = percent("Terminal growth (%)", "stocks_g", 2.5)
                p["net_debt"] = dot_number_input("Net debt", value=1_000_000_000.0, key="stocks_debt")
                p["shares"] = dot_number_input("Shares outstanding", min_value=1, value=100_000_000,
                                               step=1_000_000, key="stocks_shares")
            elif p["model"] == "DDM":
                p["d1"] = dot_number_input("Next-year dividend per share", min_value=0.0, value=2.40, key="stocks_d1")
                p["ke"] = percent("Cost of equity (%)", "stocks_ke", 10)
                p["g"] = percent("Dividend growth (%)", "stocks_g_ddm", 3)
            else:
                p["eps"] = dot_number_input("Forecast EPS", value=5.0, key="stocks_eps")
                p["pe"] = dot_number_input("Target forward P/E", min_value=0.01, value=12.0, key="stocks_pe")
            params["stocks"] = p

    if "bonds" in selected:
        with st.expander("Bonds · Fixed Income", expanded=True):
            p = {}
            p["kind"] = st.selectbox("Bond type", ["Fixed-rate bond", "Zero-coupon bond", "Floating-rate (approximation)"], key="bond_kind")
            p["nominal"] = dot_number_input("Notional", min_value=0.01, value=1000.0, key="bond_nominal")
            p["maturity"] = dot_number_input("Maturity (whole years)", min_value=1, max_value=50, value=5,
                                            step=1, key="bond_years")
            p["ytm"] = percent("YTM (%)", "bond_yield", 5)
            if p["kind"] == "Fixed-rate bond":
                p["coupon"] = percent("Annual coupon (%)", "bond_coupon", 4.5)
            elif p["kind"] == "Floating-rate (approximation)":
                p["ref"] = percent("Assumed constant reference rate (%)", "bond_ref", 4)
                p["margin"] = bps("Spread (bp)", "bond_margin", 100)
            p["market_price"] = dot_number_input("Market bond price (for YTM)", min_value=0.01,
                                                 value=980.0, key="bond_market_price")
            p["spot_raw"] = st.text_input("Spot curve (%; comma-separated; years 1, 2, ...)",
                                           "4.0, 4.2, 4.3, 4.6, 5.0", key="bond_spots")
            params["bonds"] = p

    if "options" in selected:
        with st.expander("Options · Option Pricing", expanded=True):
            p = {}
            p["model"] = st.selectbox("Model", ["Black-Scholes", "CRR", "Heston"], key="option_model")
            p["kind"] = st.selectbox("Option type", ["call", "put"], key="option_kind")
            p["valuation_date"] = st.date_input("Valuation date", value=date.today(), key="option_now")
            p["maturity_date"] = st.date_input("Expiry date", value=date.today() + timedelta(days=365),
                                                 key="option_expiry")
            p["spot"] = dot_number_input("Underlying price S0", min_value=0.01, value=100.0, key="option_spot")
            p["strike"] = dot_number_input("Strike K", min_value=0.01, value=105.0, key="option_strike")
            p["r"] = percent("Risk-free rate (%)", "option_r", 4)
            p["q"] = percent("Dividend yield (%)", "option_q", 1)
            if p["model"] in ("Black-Scholes", "CRR"):
                p["sigma"] = percent("Volatility (%)", "option_sigma", 25, min_value=0.01)
            if p["model"] == "CRR":
                p["steps"] = dot_number_input("Number of steps", min_value=5, max_value=1000, value=150,
                                              step=5, key="option_steps")
                p["exercise"] = st.selectbox("Exercise", ["european", "american"], key="option_exercise")
            if p["model"] == "Heston":
                p["v0"] = dot_number_input("Initial variance v0", min_value=0.00001,
                                           value=0.04, step=0.01, format="%.4f", key="heston_v0")
                p["kappa"] = dot_number_input("Kappa", min_value=0.01, value=2.0, key="heston_kappa")
                p["theta"] = dot_number_input("Theta", min_value=0.00001, value=0.04,
                                              step=0.01, format="%.4f", key="heston_theta")
                p["xi"] = dot_number_input("Xi", min_value=0.01, value=0.5, key="heston_xi")
                p["rho"] = st.slider("Rho", min_value=-0.99, max_value=0.99, value=-0.7,
                                      step=0.01, key="heston_rho")
                p["smile"] = st.checkbox("Generate Heston implied volatility smile (slower)",
                                          value=False, key="heston_smile")
                p["smile_n"] = st.slider("Number of strikes for volatility smile", 7, 31, 13, 2, key="heston_n")
            params["options"] = p

    if "Futures" in selected:
        with st.expander("Futures · Contract Pricing", expanded=True):
            params["Futures"] = futures_parameters("futures")

    if "FX Futures" in selected:
        with st.expander("FX Futures · Currency Futures", expanded=True):
            params["FX Futures"] = futures_parameters("fxfutures", fx_only=True)

    if "CDS" in selected:
        with st.expander("CDS · Credit Default Swaps", expanded=True):
            p = {}
            p["nominal"] = dot_number_input("CDS notional", min_value=1.0, value=1_000_000.0, key="cds_nominal")
            p["spread"] = bps("Contractual spread (bp)", "cds_spread", 100)
            p["recovery"] = percent("Recovery rate (%)", "cds_recovery", 40, min_value=0)
            p["rate"] = percent("Discount rate (%)", "cds_rate", 3)
            p["T"] = dot_number_input("Maturity (years)", min_value=0.1, value=5.0, key="cds_t")
            p["hazard_mode"] = st.selectbox("Hazard rate", ["Manual input", "Calibrated from market spread"], key="cds_hmode")
            if p["hazard_mode"] == "Manual input":
                p["hazard"] = percent("Hazard rate (%)", "cds_hazard", 2, min_value=0)
            else:
                p["market_spread"] = bps("Market CDS par spread (bp)", "cds_market", 120)
            params["CDS"] = p

    if "Swaps" in selected:
        with st.expander("IRS · Interest Rate Swaps", expanded=True):
            p = {}
            p["type"] = "IRS"
            if p["type"] == "IRS":
                p["nominal"] = dot_number_input("IRS notional", min_value=1.0, value=10_000_000.0,
                                                key="irs_nominal")
                p["fixed"] = percent("Fixed rate (%)", "irs_fixed", 5)
                p["maturity"] = dot_number_input("Maturity (years)", min_value=1, max_value=30,
                                                 value=3, key="irs_t")
                p["frequency"] = st.selectbox("Payments per year", [1, 2, 4], index=1, key="irs_frequency")
                p["side"] = st.selectbox("Position", ["payer", "receiver"], key="irs_side")
                p["curve_mode"] = st.radio("Spot curve", ["Flat", "Nodes"], horizontal=True,
                                            key="irs_curve_mode")
                if p["curve_mode"] == "Flat":
                    p["spot"] = percent("Spot zero rate (%)", "irs_spot_flat", 5)
                else:
                    n = p["maturity"] * p["frequency"]
                    defaults = ", ".join(f"{4 + j * 0.15:.2f}" for j in range(n))
                    p["spots_raw"] = st.text_area("Spot curve (%; comma-separated, one node per payment date)",
                                                    value=defaults, height=95, key="irs_spot_nodes")
            params["Swaps"] = p

    if "FRA" in selected:
        with st.expander("FRA · Forward Rate Agreements", expanded=True):
            p = {}
            p["nominal"] = dot_number_input("FRA notional", min_value=1.0, value=1_000_000.0, key="fra_nominal")
            p["fixed"] = percent("Contractual fixed rate K (%)", "fra_fixed", 5)
            p["t1"] = dot_number_input("T1 (years)", min_value=0.01, value=0.25, key="fra_t1")
            p["t2"] = dot_number_input("T2 (years)", min_value=0.02, value=0.50, key="fra_t2")
            p["alpha"] = dot_number_input("Accrual factor (year fraction)", min_value=0.001, value=0.25,
                                          format="%.4f", key="fra_alpha")
            p["side"] = st.selectbox("Position", ["receive floating", "pay floating"], key="fra_side")
            p["mode"] = st.selectbox("Mode", ["Pre-fixing valuation", "Post-fixing settlement"], key="fra_mode")
            if p["mode"] == "Pre-fixing valuation":
                p["r1"] = percent("Zero rate at T1 (%)", "fra_r1", 4.5)
                p["r2"] = percent("Zero rate at T2 (%)", "fra_r2", 5.0)
            else:
                p["fixing"] = percent("Observed reference fixing (%)", "fra_fixing", 6.0)
            params["FRA"] = p

    if "FX Forward" in selected:
        with st.expander("FX Forward · Currency Forwards", expanded=True):
            p = {}
            p["spot"] = dot_number_input("Spot (quote per base currency)", min_value=0.00001,
                                         value=4.30, key="fxfwd_spot")
            p["r_base"] = percent("Base currency EUR rate (%)", "fxfwd_eur", 3)
            p["r_quote"] = percent("Quote currency PLN rate (%)", "fxfwd_pln", 5)
            p["T"] = dot_number_input("Maturity (years)", min_value=0.01, value=1.0, key="fxfwd_t")
            p["notional"] = dot_number_input("Base currency notional", min_value=0.01, value=100_000.0,
                                             key="fxfwd_nominal")
            p["K"] = dot_number_input("Contractual exchange rate", min_value=0.00001, value=4.35,
                                     key="fxfwd_k")
            p["side"] = st.selectbox("Position", ["Buy base currency", "Sell base currency"],
                                      key="fxfwd_side")
            params["FX Forward"] = p

    if "FX Swaps" in selected:
        with st.expander("FX Swap · Currency Swaps", expanded=True):
            p = {}
            p["nominal"] = dot_number_input("Base currency notional", min_value=1.0, value=100_000.0,
                                            key="fxswap_nominal")
            p["spot"] = dot_number_input("Spot EUR/PLN", min_value=0.00001, value=4.30,
                                         key="fxswap_spot")
            p["kn"] = dot_number_input("Near rate", min_value=0.00001, value=4.30, key="fxswap_kn")
            p["kf"] = dot_number_input("Far rate", min_value=0.00001, value=4.35, key="fxswap_kf")
            p["tn"] = dot_number_input("Near maturity (years)", min_value=0.0, value=0.0,
                                       key="fxswap_tn")
            p["tf"] = dot_number_input("Far maturity (years)", min_value=0.01, value=1.0,
                                       key="fxswap_tf")
            p["re"] = percent("EUR zero rate (%)", "fxswap_re", 3)
            p["rp"] = percent("PLN zero rate (%)", "fxswap_rp", 5)
            p["side"] = st.selectbox("Near leg", ["Buy EUR", "Sell EUR"], key="fxswap_side")
            p["market_points"] = dot_number_input("Market forward points for calibration",
                                                   value=0.08687, step=0.001, format="%.5f",
                                                   key="fxswap_market_points")
            params["FX Swaps"] = p


with col2:
    st.subheader("Valuation and Risk Analytics")
    st.caption("Model prices, sensitivities, valuation drivers and scenario analysis.")
    if not selected:
        st.info("Select an instrument on the left to display its valuation and charts.")

    for instrument in selected:
        p = params.get(instrument, {})
        with st.container(border=True):
            st.markdown(f"### {DISPLAY_NAMES[instrument]}")
            try:
                if instrument == "stocks":
                    if p["model"] == "DCF (FCFF)":
                        growth = parse_percent_list(p["growth_raw"])
                        if p["estimate_wacc"]:
                            ke = stocks.capm(p["rf"], p["beta"], p["erp"])
                            wacc = stocks.wacc(ke, p["kd"], p["tax"], p["we"])
                            st.caption(f"CAPM cost of equity: {ke:.2%} · WACC: {wacc:.2%}")
                        else:
                            wacc = p["wacc"]
                        result = stocks.dcf_fcff(p["fcff0"], growth, wacc, p["g"],
                                                  p["net_debt"], p["shares"])
                        m1, m2, m3 = st.columns(3)
                        with m1: metric("DCF share price", result["price_per_share"])
                        with m2: metric("Enterprise Value", result["enterprise_value"], decimals=0)
                        with m3: metric("Upside", stocks.upside(result["price_per_share"], p["market_price"]) * 100, "%")
                        df = result["forecast"]
                        fig = go.Figure()
                        fig.add_bar(x=df["year"], y=df["fcff"], name="FCFF", marker_color=BLUE)
                        fig.add_scatter(x=df["year"], y=df["pv_fcff"], mode="lines+markers",
                                        name="PV FCFF", line=dict(color=GREEN, width=3))
                        chart(fig, "Forecast FCFF and present values")
                        table_download(df, "stocks_dcf_export", "stocks_dcf.csv")
                        wacc_shifts = [-0.005, 0, 0.005]
                        g_levels = [max(-0.5, p["g"] - 0.005), p["g"], p["g"] + 0.005]
                        if min(wacc + k for k in wacc_shifts) <= max(g_levels):
                            st.info("Sensitivity analysis requires WACC to exceed every tested terminal growth rate.")
                        else:
                            sens = stocks.dcf_sensitivity(p["fcff0"], growth, wacc,
                                                          g_levels, wacc_shifts,
                                                          p["net_debt"], p["shares"])
                            heat = go.Figure(data=go.Heatmap(
                                z=sens.to_numpy(), x=list(sens.columns), y=list(sens.index),
                                colorscale="Blues", text=np.round(sens.to_numpy(), 2),
                                texttemplate="%{text:.2f}", showscale=True))
                            chart(heat, "Share price sensitivity: WACC vs terminal growth")
                        show_inputs(p, "stocks.capm → stocks.wacc → stocks.dcf_fcff → stocks.upside / stocks.dcf_sensitivity")
                    elif p["model"] == "DDM":
                        value = stocks.ddm(p["d1"], p["ke"], p["g"])
                        a, b = st.columns(2)
                        with a: metric("DDM share price", value)
                        with b: metric("Upside", 100 * stocks.upside(value, p["market_price"]), "%")
                        yrs = np.arange(1, 11)
                        div = p["d1"] * (1 + p["g"]) ** (yrs - 1)
                        pv_div = div / (1 + p["ke"]) ** yrs
                        fig = go.Figure()
                        fig.add_bar(x=yrs, y=div, name="Dividends", marker_color=BLUE)
                        fig.add_scatter(x=yrs, y=pv_div, mode="lines+markers", name="PV of dividends",
                                        line=dict(color=GREEN, width=3))
                        chart(fig, "Dividends — 10-year profile")
                        show_inputs(p, "stocks.ddm → stocks.upside")
                    else:
                        value = stocks.pe_valuation(p["eps"], p["pe"])
                        a, b = st.columns(2)
                        with a: metric("P/E share price", value)
                        with b: metric("Upside", 100 * stocks.upside(value, p["market_price"]), "%")
                        multiples = np.linspace(max(0.5, p["pe"] - 6), p["pe"] + 6, 25)
                        line_plot(multiples, [stocks.pe_valuation(p["eps"], v) for v in multiples],
                                  "Implied share price", title="Share price vs P/E multiple", xname="P/E", yname="Share price")
                        show_inputs(p, "stocks.pe_valuation → stocks.upside")

                elif instrument == "bonds":
                    coupon = (p["coupon"] if p["kind"] == "Fixed-rate bond"
                              else 0 if p["kind"] == "Zero-coupon bond" else p["ref"] + p["margin"])
                    price = bonds.bond_price(p["ytm"], coupon, p["maturity"], p["nominal"])
                    a, b = st.columns(2)
                    with a: metric("Model price", price)
                    with b:
                        try:
                            yield_price = bonds.ytm(p["market_price"], coupon, p["maturity"], p["nominal"])
                            metric("YTM implied by market price", yield_price * 100, "%", 3)
                        except ValueError:
                            st.warning("YTM could not be found within the supported numerical range.")
                    if p["kind"] == "Floating-rate (approximation)":
                        st.warning("The floating-rate approximation assumes a constant reference rate and does not model future coupon resets.")
                    yrs = np.arange(1, p["maturity"] + 1)
                    cfs = np.full(len(yrs), p["nominal"] * coupon, dtype=float)
                    cfs[-1] += p["nominal"]
                    pv = cfs / (1 + p["ytm"]) ** yrs
                    flows = pd.DataFrame({"Year": yrs, "Cash flow": cfs, "PV": pv})
                    fig = go.Figure()
                    fig.add_bar(x=yrs, y=cfs, name="Cash flow", marker_color=BLUE)
                    fig.add_scatter(x=yrs, y=pv, name="PV", mode="lines+markers",
                                    line=dict(color=GREEN, width=3))
                    chart(fig, "Cash flows and discounted cash flows")
                    ys = np.linspace(max(-0.8, p["ytm"] - 0.04), p["ytm"] + 0.04, 41)
                    line_plot(ys * 100, [bonds.bond_price(y, coupon, p["maturity"], p["nominal"]) for y in ys],
                              "Price", title="Bond price vs YTM", xname="YTM (%)", yname="Price")
                    table_download(flows, "bond_cf_csv", "bond_cashflows.csv")
                    spots = parse_percent_list(p["spot_raw"])
                    forwards = bonds.forward_curve(spots.tolist())
                    fig = go.Figure()
                    fig.add_scatter(x=np.arange(1, len(spots) + 1), y=spots, mode="lines+markers",
                                    name="Spot", line=dict(color=BLUE, width=3))
                    fig.add_scatter(x=np.arange(1, len(spots) + 1), y=forwards, mode="lines+markers",
                                    name="Forward", line=dict(color=GREEN, width=3))
                    chart(fig, "Spot and forward curves (annual compounding)", percent_y=True)
                    show_inputs(p, "bonds.bond_price / bonds.ytm / bonds.forward_curve")

                elif instrument == "options":
                    vdate = datetime.combine(p["valuation_date"], time.min)
                    mdate = datetime.combine(p["maturity_date"], time.min)
                    if mdate <= vdate:
                        raise ValueError("The expiry date must be later than the valuation date.")
                    kwargs = dict(valuation_date=vdate, maturity_date=mdate,
                                  S0=p["spot"], K=p["strike"], r=p["r"], q=p["q"], option_type=p["kind"])
                    if p["model"] == "Black-Scholes":
                        fn = options.black_scholes
                        kwargs["sigma"] = p["sigma"]
                    elif p["model"] == "CRR":
                        fn = options.crr
                        kwargs.update(sigma=p["sigma"], steps=p["steps"], exercise=p["exercise"])
                    else:
                        fn = options.heston
                        kwargs.update(v0=p["v0"], kappa=p["kappa"], theta=p["theta"],
                                      xi=p["xi"], rho=p["rho"])
                    value = fn(**kwargs)
                    metric("Option price", value)
                    spot_bump = min(0.49 * p["spot"], max(0.0001, p["spot"] * 0.01))
                    delta = (fn(**{**kwargs, "S0": p["spot"] + spot_bump}) -
                             fn(**{**kwargs, "S0": p["spot"] - spot_bump})) / (2 * spot_bump)
                    metric("Delta (central difference)", delta, decimals=4)
                    S = np.linspace(p["spot"] * 0.7, p["spot"] * 1.3, 25)
                    prices = [fn(**{**kwargs, "S0": float(s)}) for s in S]
                    payoff = np.maximum((S - p["strike"]) if p["kind"] == "call"
                                        else (p["strike"] - S), 0)
                    fig = go.Figure()
                    fig.add_scatter(x=S, y=prices, name="Model price", line=dict(color=BLUE, width=3))
                    fig.add_scatter(x=S, y=payoff, name="Payoff at expiry", line=dict(color=GREEN, dash="dash"))
                    fig.add_vline(x=p["strike"], line_dash="dot", line_color="#64748B")
                    chart(fig, "Option price and payoff vs underlying price")
                    if p["model"] == "Heston" and p["smile"]:
                        with st.spinner("Calculating implied volatility smile using QuantLib..."):
                            smile, fig_smile = options.volatility_smile(
                                vdate, mdate, p["spot"], p["r"], p["q"], p["v0"],
                                p["kappa"], p["theta"], p["xi"], p["rho"],
                                n_strikes=p["smile_n"], option_type=p["kind"])
                        chart(fig_smile, "Heston — Implied Volatility Smile")
                        table_download(smile, "option_smile", "heston_smile.csv")
                    show_inputs(p, f"options.{fn.__name__}" + (" / options.volatility_smile" if p["model"] == "Heston" and p["smile"] else ""))

                elif instrument in ("Futures", "FX Futures"):
                    futures_results(p, instrument)

                elif instrument == "CDS":
                    if p["recovery"] >= 1 or p["recovery"] < 0:
                        raise ValueError("Recovery rate must be between 0% (inclusive) and 100% (exclusive).")
                    hazard = (p["hazard"] if p["hazard_mode"] == "Manual input"
                              else CDS.cds_calibrate(p["market_spread"], p["recovery"]))
                    report = CDS.cds_report(p["nominal"], p["spread"], p["recovery"],
                                            hazard, p["rate"], p["T"])
                    a, b, c = st.columns(3)
                    with a: metric("CDS NPV", report["npv"])
                    with b: metric("CS01 (+1 bp)", report["cs01"])
                    with c: metric("IR DV01 (+1 bp)", report["ir_dv01"])
                    st.caption(f"Survival probability: {report['survival_probability']:.2%} · "
                               f"PD: {report['default_probability']:.2%} · "
                               f"Par spread: {report['par_spread'] * 10000:.1f} bp")
                    fig = go.Figure()
                    fig.add_bar(x=["Premium Leg", "Protection Leg"],
                                y=[report["premium_leg"], report["protection_leg"]],
                                marker_color=["#94A3B8", BLUE])
                    chart(fig, "Present value of CDS legs")
                    horizon = np.linspace(0, p["T"], 80)
                    survive = CDS.survival_probability(hazard, horizon)
                    fig = go.Figure()
                    fig.add_scatter(x=horizon, y=survive, name="Survival probability",
                                    line=dict(color=BLUE, width=3))
                    fig.add_scatter(x=horizon, y=1 - survive, name="Cumulative PD",
                                    line=dict(color=RED, width=2))
                    chart(fig, "Survival and cumulative default probability curves", percent_y=True)
                    hazards = np.clip(hazard * np.array([0.5, 0.75, 1, 1.25, 1.5]), 0, None)
                    recoveries = np.clip(p["recovery"] + np.array([-0.20, -0.10, 0, 0.10, 0.20]), 0, 0.99)
                    scenarios = CDS.cds_scenarios(p["nominal"], p["spread"], p["rate"],
                                                   p["T"], hazards, recoveries)
                    pivot = scenarios.pivot(index="hazard", columns="recovery", values="NPV")
                    fig = go.Figure(data=go.Heatmap(
                        x=[f"{v:.0%}" for v in pivot.columns],
                        y=[f"{v:.2%}" for v in pivot.index], z=pivot.to_numpy(),
                        colorscale="RdBu", zmid=0, colorbar=dict(title="NPV")))
                    chart(fig, "CDS NPV — hazard rate × recovery rate")
                    table_download(scenarios, "cds_scenario_csv", "cds_scenarios.csv")
                    show_inputs({**p, "effective_hazard_rate": hazard},
                                "CDS.cds_calibrate → CDS.cds_report → CDS.cds_scenarios → CDS.survival_probability")

                elif instrument == "Swaps":
                    if p["type"] != "IRS":
                        st.warning("This type of swap is not supported.")
                    else:
                        times, alpha = swaps.payment_schedule(p["maturity"], p["frequency"])
                        if p["curve_mode"] == "Flat":
                            spots = np.full(len(times), p["spot"])
                        else:
                            spots = parse_percent_list(p["spots_raw"])
                            if len(spots) != len(times):
                                raise ValueError(f"Expected {len(times)} curve nodes, but received {len(spots)}.")
                        discount = swaps.discount_factors(spots, times)
                        forward = swaps.forward_rates(discount, alpha)
                        value = swaps.swap(p["nominal"], p["fixed"], alpha, discount,
                                           forward, side=p["side"])
                        par = swaps.par_swap_rate(alpha, discount, forward)
                        dv01 = swaps.DV01(p["nominal"], p["fixed"], spots, times, alpha, p["side"])
                        a, b, c = st.columns(3)
                        with a: metric("IRS NPV", value["npv"])
                        with b: metric("Par swap rate", par * 100, "%", 3)
                        with c: metric("DV01 (+1 bp)", dv01)
                        st.caption((f"Fixed-leg PV: {value['fixed']:,.2f} · "
                                    f"Floating-leg PV: {value['floating']:,.2f}").replace(",", " "))
                        flow = pd.DataFrame({"T": times, "alfa": alpha, "spot": spots,
                                             "df": discount, "forward": forward,
                                             "fixed_cashflow": p["nominal"] * alpha * p["fixed"],
                                             "floating_cashflow": p["nominal"] * alpha * forward})
                        fig = go.Figure()
                        fig.add_scatter(x=times, y=spots, name="Spot", line=dict(color=BLUE, width=3))
                        fig.add_scatter(x=times, y=forward, name="Forward", line=dict(color=GREEN, width=3))
                        chart(fig, "Spot curve and implied forward rates", percent_y=True)
                        table_download(flow, "irs_cf_export", "irs_cashflows.csv")
                        krdv = swaps.key_rate_dv01(p["nominal"], p["fixed"], spots, times, alpha, p["side"])
                        fig = px.bar(krdv, x="tenor", y="dv01", color_discrete_sequence=[BLUE])
                        chart(fig, "IRS Key Rate DV01 (+1 bp node shock)")
                        scen = swaps.scenario_analysis(p["nominal"], p["fixed"], spots, times, alpha,
                                                       [-100, -50, -25, 0, 25, 50, 100], p["side"])
                        line_plot(scen["shock"], scen["npv"], "NPV", title="IRS NPV under parallel curve shifts",
                                  xname="Shock (bp)", yname="NPV")
                        show_inputs({**p, "spot_rates": spots, "times": times, "alpha": alpha},
                                    "swaps.payment_schedule → swaps.discount_factors → swaps.forward_rates → swaps.swap\n"
                                    "swaps.par_swap_rate / swaps.DV01 / swaps.key_rate_dv01 / swaps.scenario_analysis")

                elif instrument == "FRA":
                    if p["t2"] <= p["t1"]:
                        raise ValueError("T2 must be greater than T1.")
                    sign = 1 if p["side"] == "receive floating" else -1
                    if p["mode"] == "Pre-fixing valuation":
                        df1 = FRA.discount_factors(p["r1"], p["t1"])
                        df2 = FRA.discount_factors(p["r2"], p["t2"])
                        F = FRA.forward_rates(df1, df2, p["alpha"])
                        npv = sign * FRA.FRA(p["nominal"], p["alpha"], F, p["fixed"], df2)
                        a, b = st.columns(2)
                        with a: metric("FRA NPV", npv)
                        with b: metric("Forward rate", F * 100, "%", 3)
                        if not np.isclose(p["alpha"], p["t2"] - p["t1"]):
                            st.warning("Risk sensitivity calculations require the accrual factor to equal T2 − T1. "
                                       "Only NPV is shown until the dates and accrual factor are consistent.")
                        else:
                            dv = sign * FRA.DV01(p["nominal"], p["fixed"], p["r1"], p["r2"], p["t1"], p["t2"])
                            metric("DV01 (+1 bp)", dv)
                            kr = FRA.key_rate_dv01(p["nominal"], p["fixed"], p["r1"],
                                                   p["r2"], p["t1"], p["t2"])
                            kr["Key Rate DV01"] *= sign
                            fig = px.bar(kr, x="Tenor", y="Key Rate DV01",
                                         color_discrete_sequence=[BLUE])
                            chart(fig, "FRA Key Rate DV01")
                            table_download(kr, "fra_key_csv", "fra_key_rate_dv01.csv")
                        st.caption("Pre-fixing FRA: simplified single-curve model, discounted to T2.")
                        show_inputs({**p, "df1": df1, "df2": df2, "forward_rate": F},
                                    "FRA.discount_factors → FRA.forward_rates → FRA.FRA / FRA.DV01 / FRA.key_rate_dv01")
                    else:
                        settlement = sign * FRA.FRA_settlement(p["nominal"], p["alpha"], p["fixing"], p["fixed"])
                        metric("Settlement at the start of the accrual period", settlement)
                        L = np.linspace(max(-0.99 / p["alpha"] + 0.1, p["fixing"] - 0.03),
                                        p["fixing"] + 0.03, 41)
                        values = [sign * FRA.FRA_settlement(p["nominal"], p["alpha"], r, p["fixed"]) for r in L]
                        line_plot(L * 100, values, "Settlement", title="FRA Settlement vs fixing rate",
                                  xname="Reference fixing (%)", yname="Settlement")
                        show_inputs(p, "FRA.FRA_settlement")

                elif instrument == "FX Forward":
                    F = FX_forwards.fx_forwards(p["spot"], p["T"], p["r_base"], p["r_quote"])
                    dfq = np.exp(-p["r_quote"] * p["T"])
                    sign = 1 if p["side"] == "Buy base currency" else -1
                    a, b, c = st.columns(3)
                    with a: metric("FX Forward Rate", F, decimals=5)
                    with b: metric("Forward points", F - p["spot"], decimals=5)
                    with c: metric("Contract NPV", sign * p["notional"] * (F - p["K"]) * dfq)
                    tenors = np.linspace(0.01, max(3, p["T"] * 1.5), 48)
                    forwards = FX_forwards.forward_curve(p["spot"], tenors,
                                                         np.full(len(tenors), p["r_base"]),
                                                         np.full(len(tenors), p["r_quote"]))
                    line_plot(tenors, forwards, "Forward", title="FX Forward curve",
                              xname="Maturity (years)", yname="EUR/PLN rate")
                    st.caption("Covered interest parity model, without cross-currency basis.")
                    show_inputs(p)

                elif instrument == "FX Swaps":
                    if p["tf"] <= p["tn"]:
                        raise ValueError("Far leg maturity must be after near leg maturity.")
                    sign = 1 if p["side"] == "Buy EUR" else -1
                    kwargs = (p["nominal"], p["spot"], p["kn"], p["kf"],
                              p["tn"], p["tf"], p["re"], p["rp"])
                    npv = sign * FX_swaps.fx_swap_npv(*kwargs)
                    delta = sign * FX_swaps.fx_delta(p["nominal"], p["tn"], p["tf"], p["re"])
                    risk = FX_swaps.fx_swap_dv01(*kwargs)
                    forward = FX_swaps.fx_forward(p["spot"], p["re"], p["rp"], p["tf"])
                    points = FX_swaps.forward_points(p["spot"], p["re"], p["rp"], p["tf"])
                    market_implied_eur = FX_swaps.calibrate_fx(p["spot"], p["market_points"], p["rp"], p["tf"])
                    a, b, c = st.columns(3)
                    with a: metric("FX Swap NPV", npv)
                    with b: metric("FX Delta", delta)
                    with c: metric("Forward points", points, decimals=5)
                    st.caption((f"EUR DV01: {sign * risk['EUR_DV01']:,.2f} · "
                                f"PLN DV01: {sign * risk['PLN_DV01']:,.2f} · "
                                f"Forward: {forward:.5f} · Implied EUR zero: {market_implied_eur:.3%}").replace(",", " "))
                    curves = pd.DataFrame({"Currency": ["EUR", "PLN"],
                                           "DV01": [sign * risk["EUR_DV01"], sign * risk["PLN_DV01"]]})
                    fig = px.bar(curves, x="Currency", y="DV01", color_discrete_sequence=[BLUE])
                    chart(fig, "Currency interest-rate sensitivities")
                    scenarios = FX_swaps.fx_scenarios(*kwargs, [-0.05, -0.02, 0, 0.02, 0.05])
                    scenarios["npv"] *= sign
                    line_plot(scenarios["spot"], scenarios["npv"], "NPV",
                              title="FX Swap NPV vs spot rate", xname="Spot EUR/PLN", yname="NPV")
                    table_download(scenarios, "fxswap_scenario_csv", "fxswap_scenarios.csv")
                    show_inputs(p, "FX_swaps.fx_swap_npv / FX_swaps.fx_delta / FX_swaps.fx_swap_dv01\n"
                                "FX_swaps.fx_forward / FX_swaps.forward_points / FX_swaps.calibrate_fx / FX_swaps.fx_scenarios")

            except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError,
                    ImportError, RuntimeError, KeyError, IndexError) as exc:
                message = str(exc)
                if isinstance(exc, ImportError) and "Model Hestona wymaga pakietu QuantLib" in message:
                    message = "Heston pricing requires QuantLib. Install the QuantLib package to continue."
                if ".py" in message or "backend" in message.lower():
                    message = "An internal calculation error occurred. Check the input parameters."
                st.error(f"Valuation failed: {message}")
                with st.expander("Review input assumptions"):
                    st.json(p)
