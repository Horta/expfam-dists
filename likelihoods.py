from numpy import linspace
from numpy import exp
from numpy import sqrt

import scipy
import scipy.stats as st

from infra import LikNorm
from infra import Moments
from binomial import get_binomials
from bernoulli import get_bernoullis
from normal import get_normals
from exponential import get_exponentials
from poisson import get_poissons
from gamma import get_gammas
from geometric import get_geometrics


class TestCase(object):

    def __init__(self, lik_name, lik_params, normal_mean, normal_variance,
                 log_zeroth, mean, variance):
        self.normal_mean = normal_mean
        self.normal_variance = normal_variance
        self.lik_name = lik_name
        self.lik_params = lik_params
        self.log_zeroth = log_zeroth
        self.mean = mean
        self.variance = variance


def get_prior_normals():
    means = linspace(-1, +1, 5)
    variances = linspace(1e-1, 1, 5)
    normals = []
    for m in means:
        for variance in variances:
            normals += [st.norm(loc=m, scale=sqrt(variance))]
    return normals


def apply_liknorm(liks, normals, a, b):
    test_cases = []
    for normal in normals:
        for lik in liks:
            try:
                m = Moments(LikNorm(lik, normal.mean(), normal.var()), a, b)
            except Exception as e:
                print(e)
                continue
            t = TestCase(lik.name, lik.params, normal.mean(), normal.var(),
                         m.log_zeroth, m.mean, m.variance)
            test_cases += [t]
    return test_cases


def save_test_cases(test_cases, name):
    import pandas as pd
    param_names = test_cases[0].lik_params.keys()
    df = pd.DataFrame(columns=['normal_mean', 'normal_variance', 'likname']
                      + param_names + ['log_zeroth', 'mean', 'variance'])
    for i in range(len(test_cases)):
        t = test_cases[i]
        df.loc[i] = [t.normal_mean, t.normal_variance, t.lik_name] + \
            [t.lik_params[param_names[j]] for j in range(len(param_names))] + \
            [t.log_zeroth, t.mean, t.variance]

    df.to_csv("/Users/horta/workspace/liknorm/test/table_%s.csv" %
              name, float_format="%.17g", index=False)

if __name__ == '__main__':
    normals = get_prior_normals()
    # save_test_cases(apply_liknorm(get_exponentials(), normals, -500, -1e-5),
    #                 "exponential")
    # save_test_cases(apply_liknorm(get_geometrics(), normals, -500, -1e-5),
    #                 "geometric")
    # save_test_cases(apply_liknorm(get_binomials(), normals, -1000, 1000),
    #                 "binomial")
    save_test_cases(apply_liknorm(get_bernoullis(), normals, -1000, 1000),
                    "bernoulli")
    save_test_cases(apply_liknorm(get_poissons(), normals, -1000, 1000),
                    "poisson")
    save_test_cases(apply_liknorm(get_gammas(), normals, -500, -1e-5),
                    "gamma")
