import scipy.stats as st

from numpy import exp
from numpy import sqrt
from numpy import linspace


def get_normals():
    means = linspace(-2, +2, 5)
    variances = linspace(1e-2, 1, 5)

    class Lik(object):

        def __init__(self, mean, variance):
            self._mean = mean
            self._variance = variance
            self.name = "normal"
            self.params = dict(mean=mean, variance=variance)

        def _canonical(self, x):
            return x

        def __call__(self, x):
            p = self._canonical(x)
            return st.norm(loc=self._mean, scale=sqrt(self._variance)).pdf(x)

    liks = []
    for mean in means:
        for variance in variances:
            liks += [Lik(mean, variance)]
    return liks
