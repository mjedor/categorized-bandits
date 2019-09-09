from scipy.stats import truncnorm
from random import gauss
from math import sqrt


class Gaussian:
    """ Gaussian distributed arm """

    def __init__(self, mu, sigma2):
        self.mu = mu
        self.sigma2 = sigma2
        self.expectation = mu
        
    def draw(self):
        return gauss(self.mu, sqrt(self.sigma2))
