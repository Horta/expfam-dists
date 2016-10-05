from __future__ import division

from collections import OrderedDict

import scipy.stats as st
from scipy.special import gamma
from scipy.special import gammaln

from numpy import exp
from numpy import log
from numpy import sqrt
from numpy import linspace
from numpy import clip


class Lik(object):

    def __init__(self, a, x):
        self.name = "gamma"
        self._a = a
        self._x = x
        self.params = OrderedDict([('x', x), ('a', a)])

    def _canonical(self, x):
        return -1 / x

    def __call__(self, x):
        s = self._canonical(x)
        a = self._a
        y = self._x
        return exp((a - 1) * log(a * y) - a * y / s + log(a) - gammaln(a) - a * log(s))

    def fory(self, y, s):
        a = self._a
        return exp((a - 1) * log(a * y) - a * y / s + log(a) - gammaln(a) - a * log(s))


def get_gammas():

    as_ = [0.5, 1.0, 1.5]
    ys = [0.5, 1.0, 5.0]

    liks = []
    for a in as_:
        for y in ys:
            liks += [Lik(a, y)]
    return liks


def impl(y, theta, aphi):
    a = 1 / aphi
    s = -1 / theta
    c = a * log(a) + (a - 1) * log(y) - gammaln(a)
    return exp((y * theta - log(-1 / theta)) / aphi + c)

if __name__ == '__main__':
    a = 0.6

    def create_fory(m, s):
        def func(y):
            return y**m * Lik(a, y).fory(y, s)
        return func

    from scipy.integrate import quad
    s = 1.5
    print(1 - quad(create_fory(0, s), 1e-7, 5000, limit=500)[0])
    print(s - quad(create_fory(1, s), 1e-7, 5000, limit=500)[0])
    print(s * s / a - (quad(create_fory(2, s), 1e-7, 5000, limit=500)
                       [0] - quad(create_fory(1, s), 1e-7, 5000, limit=500)[0]**2))

    print(1 - quad(lambda y: impl(y, theta=-1 / s,
                                  aphi=1 / a), 1e-7, 5000, limit=500)[0])
    print(s - quad(lambda y: y * impl(y, theta=-1 /
                                      s, aphi=1 / a), 1e-7, 5000, limit=500)[0])
    print(s * s / a - quad(lambda y: y * y * impl(y, theta=-1 / s, aphi=1 / a), 1e-7, 5000, limit=500)
          [0] + quad(lambda y: y * impl(y, theta=-1 / s, aphi=1 / a), 1e-7, 5000, limit=500)[0]**2)
