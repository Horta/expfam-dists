from numpy import sqrt
from numpy import log

import scipy.stats as st
from scipy.integrate import quad


class LikNorm(object):

    def __init__(self, likelihood, mean, variance):
        self._likelihood = likelihood
        self._normal_pdf = st.norm(loc=mean, scale=sqrt(variance)).pdf
        self.mean = mean
        self.variance = variance

    def joint(self, x, m):
        return x**m * self._likelihood(x) * self._normal_pdf(x)


class Moments(object):

    def __init__(self, liknorm, min_, max_):
        a = max(liknorm.mean - 7 * sqrt(liknorm.variance), min_)
        b = min(liknorm.mean + 7 * sqrt(liknorm.variance), max_)

        if liknorm.joint(a, 0) < 1e-64 or liknorm.joint(b, 0) < 1e-64:
            msg = "Z points too small: %g %g."\
                % (liknorm.joint(a, 0), liknorm.joint(b, 0))
            msg += " For left and right: %g %g." % (a, b)
            msg += " For y and aphi: %g %g." % (liknorm._likelihood.params['y'],
                                                liknorm._likelihood.params['aphi'])
            print(msg)
            raise Exception(msg)
        else:
            print("WORKED")

        Z = quad(lambda x: liknorm.joint(x, 0), a, b, limit=500,
                 epsabs=1e-12, epsrel=1e-12)[0]
        umom1 = quad(lambda x: liknorm.joint(x, 1), a, b, limit=500)[0]
        mean = umom1 / Z
        umom2 = quad(lambda x: liknorm.joint(x, 2), a, b, limit=500)[0]
        mom2 = umom2 / Z
        variance = mom2 - mean * mean

        self.log_zeroth = log(Z)
        self.mean = mean
        self.variance = variance
