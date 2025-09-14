# Project 8 — Portfolio Optimization (Mean–Variance, Risk Parity)

**Goal:** Construct optimal portfolios on synthetic assets with:
- Global minimum variance
- Max Sharpe (with risk-free rate)
- Naive risk parity (equal risk contribution) via iterative solver

## Structure
```
Project8_Portfolio_Optimization/
  ├─ data/
  │   └─ returns.csv
  ├─ outputs/
  ├─ src/
  │   ├─ generate_returns.py
  │   ├─ mean_variance.py
  │   └─ risk_parity.py
  ├─ requirements.txt
  └─ README.md
```

## Quickstart
```bash
pip install -r requirements.txt
python src/generate_returns.py
python src/mean_variance.py
python src/risk_parity.py
```
