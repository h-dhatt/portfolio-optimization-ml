import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rets = pd.read_csv(root/'data'/'returns.csv', index_col=0, parse_dates=True)
mu = rets.mean().values
Sigma = rets.cov().values
n = len(mu)

def port_stats(w, rf=0.0):
    w = np.array(w)
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    sharpe = (ret - rf) / (vol + 1e-12)
    return ret, vol, sharpe

def min_var():
    cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0},)
    bnds = tuple((0.0,1.0) for _ in range(n))
    x0 = np.ones(n)/n
    obj = lambda w: w @ Sigma @ w
    res = minimize(obj, x0, bounds=bnds, constraints=cons)
    return res.x

def max_sharpe(rf=0.0001):
    cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0},)
    bnds = tuple((0.0,1.0) for _ in range(n))
    x0 = np.ones(n)/n
    obj = lambda w: -port_stats(w, rf)[2]
    res = minimize(obj, x0, bounds=bnds, constraints=cons)
    return res.x

w_minvar = min_var()
w_maxsharpe = max_sharpe()

out = root/'outputs'
out.mkdir(exist_ok=True)
pd.Series(w_minvar, index=rets.columns).to_csv(out/'weights_minvar.csv')
pd.Series(w_maxsharpe, index=rets.columns).to_csv(out/'weights_maxsharpe.csv')

# Efficient frontier (grid over target return)
targets = np.linspace(mu.min(), mu.max(), 40)
vols = []
for tr in targets:
    cons = (
        {'type':'eq','fun': lambda w, tr=tr: w@mu - tr},
        {'type':'eq','fun': lambda w: np.sum(w)-1.0},
    )
    bnds = tuple((0.0,1.0) for _ in range(n))
    x0 = np.ones(n)/n
    obj = lambda w: w @ Sigma @ w
    res = minimize(obj, x0, bounds=bnds, constraints=cons)
    vols.append(np.sqrt(res.x @ Sigma @ res.x))

plt.figure()
plt.plot(vols, targets)
plt.xlabel('Volatility'); plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.tight_layout()
plt.savefig(out/'efficient_frontier.png', dpi=150)
print('Saved portfolio weights and frontier plot.')