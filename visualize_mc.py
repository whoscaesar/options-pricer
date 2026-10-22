"""
Visualize how the Monte Carlo pricer works, in three panels:
1. Simulated stock price paths over one year
2. Distribution of final prices and which scenarios pay off
3. Convergence of the Monte Carlo price to Black-Scholes
"""

import numpy as np
import matplotlib.pyplot as plt
from black_scholes import bs_price
from monte_carlo import mc_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
rng = np.random.default_rng(42)

BLUE, ORANGE, GREY, INK = "#2a6fdb", "#e8833a", "#9aa0a6", "#202124"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
fig.suptitle("Monte Carlo option pricing: European call, S=100, K=100, T=1, r=5%, σ=20%",
             fontsize=13, color=INK)

# 1. Simulated paths
ax = axes[0]
n_paths, n_steps = 50, 252
dt = T / n_steps
Z = rng.standard_normal((n_paths, n_steps))
log_steps = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
paths = S * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)]))
t = np.linspace(0, T, n_steps + 1)
for p in paths:
    ax.plot(t, p, color=BLUE if p[-1] > K else GREY, lw=0.8, alpha=0.7)
ax.axhline(K, color=ORANGE, lw=1.5, ls="--", label="Strike K = 100")
ax.set_title("1. Simulate 50 possible stock paths", loc="left")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Stock price (€)")
ax.legend(frameon=False)
ax.text(0.02, 0.97, "blue = ends above strike (call pays off)\ngrey = ends below (worth 0)",
        transform=ax.transAxes, va="top", fontsize=9, color=INK)

# 2. Distribution of final prices
ax = axes[1]
ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * rng.standard_normal(100_000))
bins = np.linspace(40, 200, 81)
ax.hist(ST[ST <= K], bins=bins, color=GREY, label="Payoff = 0")
ax.hist(ST[ST > K], bins=bins, color=BLUE, label="Payoff = S_T − K")
ax.axvline(K, color=ORANGE, lw=1.5, ls="--")
share = (ST > K).mean()
ax.set_title("2. Final prices of 100,000 simulations", loc="left")
ax.set_xlabel("Stock price at maturity (€)")
ax.set_ylabel("Number of simulations")
ax.legend(frameon=False)
ax.text(0.98, 0.97, f"{share:.0%} of scenarios pay off\naverage payoff, discounted\n= option price",
        transform=ax.transAxes, va="top", ha="right", fontsize=9, color=INK)

# 3. Convergence
ax = axes[2]
bs = bs_price(S, K, T, r, sigma, "call")
ns = np.unique(np.logspace(2, 6, 25).astype(int))
prices, ses = zip(*[mc_price(S, K, T, r, sigma, "call", n_sims=n, seed=int(n)) for n in ns])
prices, ses = np.array(prices), np.array(ses)
ax.fill_between(ns, prices - 1.96 * ses, prices + 1.96 * ses, color=BLUE, alpha=0.15,
                label="95% confidence band")
ax.plot(ns, prices, color=BLUE, marker="o", ms=3, lw=1.5, label="Monte Carlo price")
ax.axhline(bs, color=ORANGE, lw=1.5, ls="--", label=f"Black-Scholes = {bs:.2f}")
ax.set_xscale("log")
ax.set_title("3. More simulations → closer to Black-Scholes", loc="left")
ax.set_xlabel("Number of simulations (log scale)")
ax.set_ylabel("Call price (€)")
ax.legend(frameon=False, loc="lower right")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("monte_carlo.png", dpi=150, bbox_inches="tight")
print("Saved monte_carlo.png")
