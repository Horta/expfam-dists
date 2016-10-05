from __future__ import division
import scipy.stats as st

from numpy import exp
from numpy import sqrt


def get_bernoullis():
    K = [0, 1]

    class Lik(object):

        def __init__(self, K):
            self._K = K
            self.name = "bernoulli"
            self.params = dict(k=K)

        def _canonical(self, x):
            return 1 / (1 + exp(-x))

        def __call__(self, x):
            p = self._canonical(x)
            return st.bernoulli(p=p).pmf(self._K)

    return [Lik(0), Lik(1)]
