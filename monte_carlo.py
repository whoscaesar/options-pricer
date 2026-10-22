"""
Phase 2: Monte Carlo pricer for European options.

Idea: simulate many possible stock prices at maturity, compute the option
payoff in each scenario, average them, and discount back to today.
The result should converge to the Black-Scholes price.
"""

import numpy as np
from black_scholes import bs_price


def simulate_final_prices(S, T, r, sigma, n_sims, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def mc_price(S, K, T, r, sigma, option_type="call", n_sims=100_000, seed=None):
    ST = simulate_final_prices(S, T, r, sigma, n_sims, seed)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(n_sims)
    return price, std_error


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    bs_call = bs_price(S, K, T, r, sigma, "call")
    print(f"Black-Scholes call price: {bs_call:.4f}\n")

    print(f"{'Simulations':>12} {'MC price':>10} {'Std error':>10} {'Error vs BS':>12}")
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        price, se = mc_price(S, K, T, r, sigma, "call", n_sims=n, seed=42)
        print(f"{n:>12,} {price:>10.4f} {se:>10.4f} {price - bs_call:>12.4f}")
