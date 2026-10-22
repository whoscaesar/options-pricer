"""
Phase 1: Black-Scholes pricer for European options, with Greeks.

Inputs:
    S     spot price of the underlying
    K     strike price
    T     time to maturity in years (e.g. 0.5 = six months)
    r     risk-free rate, annual, continuous (e.g. 0.03 = 3%)
    sigma volatility, annual (e.g. 0.20 = 20%)
"""

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call"):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100  # per 1 percentage point change in vol

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # per day
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    print(f"Call price: {call:.4f}")
    print(f"Put price:  {put:.4f}")

    # Sanity check: put-call parity says C - P = S - K*e^(-rT)
    parity_gap = (call - put) - (S - K * np.exp(-r * T))
    print(f"Put-call parity gap (should be ~0): {parity_gap:.2e}")

    print("\nCall Greeks:")
    for name, value in bs_greeks(S, K, T, r, sigma, "call").items():
        print(f"  {name:6s} {value:.4f}")
