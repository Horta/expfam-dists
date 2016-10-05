from __future__ import division
import scipy.stats as st

from numpy import exp
from numpy import sqrt


def get_poissons():
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    class Lik(object):

        def __init__(self, k):
            self.name = "poisson"
            self._k = k
            self.params = dict(k=k)

        def __call__(self, x):
            return st.poisson(exp(x)).pmf(self._k)

    return [Lik(yi) for yi in y]
