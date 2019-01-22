import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as st

def find_optimal_poisson(x):
    def create_cost(x):
        def cost(alpha):
            mu = alpha[0]
            return -st.poisson(mu=mu).logpmf(x).sum()
        return cost

    x = np.asarray(x, float)
    x -= x.min()
    x0 = np.asarray([x.mean()])
    r = sp.optimize.fmin_l_bfgs_b(create_cost(x), x0,
                              bounds=[(1e-4, 10000.0)],
                              approx_grad=True)
    return r

def poisson_similarity(x):
    try:
        x = np.asarray(x, float)
    except ValueError:
        return 0.
    if not is_discrete(x):
        return 0.0
    r = find_optimal_poisson(x)
    if r[2]['warnflag'] != 0:
        print(r)
    return st.kstest(x, st.poisson(mu=r[0][0]).cdf).statistic

def is_discrete(x):
    return np.all(np.asarray(x, float) == np.asarray(x, int))