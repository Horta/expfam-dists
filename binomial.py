from __future__ import division
import scipy.stats as st

from numpy import exp
from numpy import sqrt


def get_binomials():
    N = [1, 2, 3, 4, 5]
    K = []
    for n in N:
        K += [[k for k in range(n + 1)]]

    class Lik(object):

        def __init__(self, N, K):
            self._N = N
            self._K = K
            self.name = "binomial"
            self.params = dict(k=K, n=N)

        def _canonical(self, x):
            return 1 / (1 + exp(-x))

        def __call__(self, x):
            p = self._canonical(x)
            return st.binom(n=self._N, p=p).pmf(k=self._K)

    liks = []
    for i in range(len(N)):
        for k in K[i]:
            liks += [Lik(N[i], k)]
    return liks
