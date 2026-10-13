
import numpy as np
import pandas as pd


def capm(risk_free_rate, beta, market_risk_premium):
    return risk_free_rate + beta * market_risk_premium


def wacc(cost_equity, cost_debt, tax_rate, equity_weight):
    if not 0 <= equity_weight <= 1 or not 0 <= tax_rate <= 1:
        raise ValueError("Weights and tax rate must be between 0 and 1.")

    debt_weight = 1 - equity_weight

    return (equity_weight * cost_equity
            + debt_weight * cost_debt * (1 - tax_rate))


def dcf_fcff(fcff0, growth_rates, discount_rate,
             terminal_growth, net_debt, shares):

    growth_rates = np.asarray(growth_rates, dtype=float)

    if growth_rates.ndim != 1 or growth_rates.size == 0:
        raise ValueError("Provide a list of annual growth rates.")

    if shares <= 0 or discount_rate <= terminal_growth or discount_rate <= -1:
        raise ValueError("Invalid shares or discount rates.")

    years = np.arange(1, len(growth_rates) + 1)

    fcff = fcff0 * np.cumprod(1 + growth_rates)
    pv_fcff = fcff / (1 + discount_rate) ** years

    terminal_value = (
        fcff[-1] * (1 + terminal_growth)
        / (discount_rate - terminal_growth)
    )

    pv_terminal = terminal_value / (1 + discount_rate) ** len(years)

    enterprise_value = float(np.sum(pv_fcff) + pv_terminal)
    equity_value = enterprise_value - net_debt
    price = equity_value / shares

    forecast = pd.DataFrame({
        "year": years,
        "growth": growth_rates,
        "fcff": fcff,
        "pv_fcff": pv_fcff
    })

    return {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "price_per_share": price,
        "pv_terminal": float(pv_terminal),
        "forecast": forecast
    }


def ddm(dividend_next_year, cost_equity, dividend_growth):
    if cost_equity <= dividend_growth:
        raise ValueError("Cost of equity must exceed dividend growth.")

    return dividend_next_year / (cost_equity - dividend_growth)


def pe_valuation(eps_next_year, target_forward_pe):
    if eps_next_year <= 0 or target_forward_pe <= 0:
        raise ValueError("Forward EPS and P/E must be positive.")

    return eps_next_year * target_forward_pe


def upside(estimated_price, market_price):
    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    return estimated_price / market_price - 1


def dcf_sensitivity(fcff0, growth_rates, base_wacc,
                    growth_levels, wacc_changes, net_debt, shares):

    results = []

    for shift in wacc_changes:
        current_wacc = base_wacc + shift
        row = {"WACC": f"{current_wacc:.2%}"}

        for g in growth_levels:
            price = dcf_fcff(
                fcff0, growth_rates, current_wacc,
                g, net_debt, shares
            )["price_per_share"]

            row[f"g={g:.1%}"] = round(price, 2)

        results.append(row)

    return pd.DataFrame(results).set_index("WACC")