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
        self.name = "geometric"
        self._x = x
        self.params = dict(x=x)

    def _canonical(self, x):
        return exp(x) / (1 - exp(x))

    def __call__(self, x):
        mean = self._canonical(x)
        p = 1 / (1 + mean)
        return exp(self._x * log(1 - p) + log(p))


def get_geometrics():

    ys = [0.5, 1.0, 5.0]

    liks = []
    for y in ys:
        liks += [Lik(y)]
    return liks
