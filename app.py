"""
Streamlit app: interactive options pricer.
Run locally with:  streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from black_scholes import bs_price, bs_greeks
from monte_carlo import mc_price

BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#9aa0a6"

st.set_page_config(page_title="Options Pricer", layout="wide")
st.title("Options Pricer")
st.caption("European options priced with Black-Scholes and Monte Carlo simulation.")

# ---------- Inputs ----------
with st.sidebar:
    st.header("Inputs")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True)
    S = st.slider("Spot price S (€)", 50.0, 150.0, 100.0, 1.0)
    K = st.slider("Strike K (€)", 50.0, 150.0, 100.0, 1.0)
    T = st.slider("Time to maturity T (years)", 0.05, 3.0, 1.0, 0.05)
    r_pct = st.slider("Risk-free rate r (%)", 0.0, 10.0, 5.0, 0.25)
    sigma_pct = st.slider("Volatility σ (%)", 5.0, 80.0, 20.0, 1.0)
    st.divider()
    n_sims = st.select_slider("Monte Carlo simulations",
                              options=[1_000, 10_000, 100_000, 500_000], value=100_000)

r, sigma = r_pct / 100, sigma_pct / 100

# ---------- Price and Greeks ----------
price = bs_price(S, K, T, r, sigma, option_type)
mc, se = mc_price(S, K, T, r, sigma, option_type, n_sims=n_sims, seed=42)
greeks = bs_greeks(S, K, T, r, sigma, option_type)

c1, c2, c3 = st.columns(3)
c1.metric("Black-Scholes price", f"€{price:.4f}")
c2.metric("Monte Carlo price", f"€{mc:.4f}", f"{mc - price:+.4f} vs BS", delta_color="off")
c3.metric("MC 95% interval", f"€{mc - 1.96 * se:.2f} to €{mc + 1.96 * se:.2f}")

st.subheader("Greeks")
g = st.columns(5)
labels = {
    "delta": ("Delta", "price change per €1 move in S"),
    "gamma": ("Gamma", "delta change per €1 move in S"),
    "vega": ("Vega", "price change per +1% volatility"),
    "theta": ("Theta", "price change per day"),
    "rho": ("Rho", "price change per +1% rate"),
}
for col, (key, (name, help_text)) in zip(g, labels.items()):
    col.metric(name, f"{greeks[key]:.4f}", help=help_text)

# ---------- Charts ----------
tab1, tab2, tab3 = st.tabs(["Payoff at maturity", "Price vs spot", "Monte Carlo paths"])

with tab1:
    s_range = np.linspace(max(1, K * 0.4), K * 1.6, 300)
    payoff = np.maximum(s_range - K, 0) if option_type == "call" else np.maximum(K - s_range, 0)
    profit = payoff - price
    breakeven = K + price if option_type == "call" else K - price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_range, y=payoff, name="Payoff", line=dict(color=GREY, dash="dot")))
    fig.add_trace(go.Scatter(x=s_range, y=profit, name="Profit (payoff − premium)",
                             line=dict(color=BLUE, width=3)))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.add_vline(x=K, line_color=ORANGE, line_dash="dash", annotation_text=f"Strike {K:.0f}")
    fig.add_vline(x=breakeven, line_color=BLUE, line_dash="dot",
                  annotation_text=f"Break-even {breakeven:.2f}", annotation_position="bottom right")
    fig.update_layout(xaxis_title="Stock price at maturity (€)", yaxis_title="€",
                      height=450, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Maximum loss is the premium (€{price:.2f}). "
               f"The option only makes a profit beyond the break-even of €{breakeven:.2f}.")

with tab2:
    s_range = np.linspace(max(1, K * 0.5), K * 1.5, 200)
    fig = go.Figure()
    for t_val, color, name in [(T, BLUE, f"T = {T:.2f} years"), (1e-6, GREY, "At maturity")]:
        prices = [bs_price(s, K, t_val, r, sigma, option_type) for s in s_range]
        fig.add_trace(go.Scatter(x=s_range, y=prices, name=name, line=dict(color=color)))
    fig.add_trace(go.Scatter(x=[S], y=[price], mode="markers", name="Current",
                             marker=dict(color=ORANGE, size=10)))
    fig.update_layout(xaxis_title="Spot price S (€)", yaxis_title="Option price (€)",
                      height=450, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption("The gap between the blue line and the grey line is the time value of the option. "
               "The slope of the blue line at the orange dot is delta.")

with tab3:
    rng = np.random.default_rng(42)
    n_paths, n_steps = 60, 252
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    steps = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)]))
    t_axis = np.linspace(0, T, n_steps + 1)
    fig = go.Figure()
    for p in paths:
        in_money = p[-1] > K if option_type == "call" else p[-1] < K
        fig.add_trace(go.Scatter(x=t_axis, y=p, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=BLUE if in_money else GREY, width=1), opacity=0.6))
    fig.add_hline(y=K, line_color=ORANGE, line_dash="dash", annotation_text=f"Strike {K:.0f}")
    fig.update_layout(xaxis_title="Time (years)", yaxis_title="Stock price (€)",
                      height=450, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each line is one simulated future. Blue paths end in the money, grey paths expire "
               "worthless. The option price is the discounted average payoff over all simulations.")

st.divider()
st.caption("Coming next: binomial tree for American options, and implied volatility from live market data.")
