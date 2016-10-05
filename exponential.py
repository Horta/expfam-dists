from __future__ import division

import scipy.stats as st
from scipy.special import gamma
from scipy.special import gammaln

from numpy import exp
from numpy import log
from numpy import sqrt
from numpy import linspace
from numpy import clip


class Lik(object):

    def __init__(self, x):
        self.name = "exponential"
        self._x = x
        self.params = dict(x=x)

    def _canonical(self, x):
        return -1 / x

    def __call__(self, x):
        lam = 1 / self._canonical(x)
        return lam * exp(- self._x * lam)


def get_exponentials():

    ys = [0.5, 1.0, 5.0]

    liks = []
    for y in ys:
        liks += [Lik(y)]
    return liks
