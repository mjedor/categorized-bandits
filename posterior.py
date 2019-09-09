from numpy.random import normal
from math import sqrt


class ImproperGaussian:
    """ Improper Gaussian prior """
    def __init__(self):
        self.cum_reward = 0
        self.n = 0

    def reset(self):
        self.cum_reward = 0
        self.n = 0

    def update(self, obs):
        self.n += 1
        self.cum_reward += obs

    def sample(self):
        return normal(self.cum_reward / self.n, 1 / sqrt(self.n))
