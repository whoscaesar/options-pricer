# Options Pricer

Options pricing simulator in Python, built step by step.

## Features

- [x] Black-Scholes pricing for European calls and puts
- [x] Greeks: delta, gamma, vega, theta, rho
- [ ] Monte Carlo pricer
- [ ] Binomial tree (American options)
- [ ] Implied volatility and volatility smile on live market data
- [ ] Interactive Streamlit app

## Run it

```bash
pip install -r requirements.txt
python black_scholes.py
```

Example output (S=100, K=100, T=1, r=5%, sigma=20%):

```
Call price: 10.4506
Put price:  5.5735
Put-call parity gap (should be ~0): 0.00e+00
```

## The model

Black-Scholes prices a European call as

C = S·N(d1) − K·e^(−rT)·N(d2)

with d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T) and d2 = d1 − σ√T.
The put follows from put-call parity: C − P = S − K·e^(−rT).
