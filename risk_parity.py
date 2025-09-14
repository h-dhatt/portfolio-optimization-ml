import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize

root = Path(__file__).resolve().parents[1]
rets = pd.read_csv(root/'data'/'returns.csv', index_col=0, parse_dates=True)
Sigma = rets.cov().values
n = Sigma.shape[0]

def risk_contrib(w, Sigma):
    w = np.asarray(w)
    port_var = w @ Sigma @ w
    mrc = Sigma @ w  # marginal risk contributions
    rc = w * mrc
    return rc, port_var

def objective(w):
    rc, port_var = risk_contrib(w, Sigma)
    target = port_var / n
    return ((rc - target)**2).sum()

cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0},)
bnds = tuple((0.0,1.0) for _ in range(n))
x0 = np.ones(n)/n
res = minimize(objective, x0, bounds=bnds, constraints=cons)
w = res.x

out = root/'outputs'
out.mkdir(exist_ok=True)
pd.Series(w, index=rets.columns).to_csv(out/'weights_risk_parity.csv')
print('Saved risk parity weights.')