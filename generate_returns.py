import numpy as np, pandas as pd
from pathlib import Path

def synth_returns(T=1000, N=6, seed=7):
    rng = np.random.default_rng(seed)
    # generate correlated normals via random covariance
    A = rng.normal(size=(N,N))
    cov = A @ A.T
    cov /= np.max(np.diag(cov))
    mu = rng.normal(0.0005, 0.001, size=N)  # daily drift
    R = rng.multivariate_normal(mean=mu, cov=cov, size=T)
    dates = pd.bdate_range('2018-01-01', periods=T)
    df = pd.DataFrame(R, index=dates, columns=[f'A{i+1}' for i in range(N)])
    return df

if __name__ == '__main__':
    df = synth_returns()
    out = Path(__file__).resolve().parents[1] / 'data' / 'returns.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f'Wrote {out}, shape={df.shape}')