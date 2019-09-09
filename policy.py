from numpy.random import choice
import numpy as np
from math import sqrt, log
from numpy.linalg import norm
from scipy.optimize import minimize


MAX_SAMPLE = 100 # Maximum number of posterior sampling in the Murphy Sampling algorithm


class IndexPolicy:
    """ Class that implements a generic index policy """
    def __init__(self, nb_arms):
        self.nb_arms = nb_arms
        
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        
    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
    
    def choice(self):
        """ In an index policy, choose at random an arm with maximal index """

        if self.t <= self.nb_arms:
            return self.t - 1

        index = [self.compute_index(arm) for arm in range(self.nb_arms)]
        return choice(np.flatnonzero(index == np.amax(index)))
    
    def get_reward(self, arm, reward):
        self.nb_draws[arm] += 1
        self.cum_reward[arm] += reward
        self.t += 1

    
class UCB(IndexPolicy):
    """ UCB algorithm """
    
    def __init__(self, nb_arms, c=2., amplitude=1., lower=0.):
        super().__init__(nb_arms)
        self.c = c

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.cum_reward[arm] / self.nb_draws[arm] + sqrt(self.c * log(self.t) / self.nb_draws[arm])
        
        
class TS(IndexPolicy):
    """ Thompson sampling algorithm """

    def __init__(self, nb_arms, posterior):
        self.nb_arms = nb_arms
        self.t = 1
        self.posterior = dict()
        for arm in range(self.nb_arms):
            self.posterior[arm] = posterior()

    def start_game(self):
        self.t = 1
        for arm in range(self.nb_arms):
            self.posterior[arm].reset()

    def get_reward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.t += 1

    def compute_index(self, arm):
        return self.posterior[arm].sample()

    
class MS:
    """ Generic Murphy sampling algorithm """
    
    def __init__(self, nb_categories, nb_arms, posterior):
        self.nb_categories = nb_categories
        if not isinstance(nb_arms, list):
            self.nb_arms = [nb_arms] * nb_categories
        else:
            self.nb_arms = nb_arms

        self.t = 1
        self.posterior = [[posterior() for _ in range(self.nb_arms[i])] for i in range(nb_categories)]

    def start_game(self):
        self.t = 1
        for category in range(self.nb_categories):
            for arm in range(self.nb_arms[category]):
                self.posterior[category][arm].reset()

    def get_reward(self, category, arm, reward):
        #reward_bernoulli = float(random() < reward)
        self.posterior[category][arm].update(reward)
        self.t += 1

    def compute_index(self, category, arm):
        return self.posterior[category][arm].sample()

    def choice(self):
        # Draw each arm once
        if self.t <= self.nb_categories * self.nb_arms[0]:
            return (self.t - 1) // self.nb_arms[0], (self.t - 1) % self.nb_arms[0]

        # Compute means
        cond = False
        means = [[0. for _ in range(self.nb_arms[i])] for i in range(self.nb_categories)]
        cpt = 0
        while not cond and cpt < MAX_SAMPLE:
            # Sample
            for category in range(self.nb_categories):
                for arm in range(self.nb_arms[category]):
                    means[category][arm] = self.compute_index(category, arm)

            # Verify condition
            cond, choice_category = self.verify_dominance(means)
            
            cpt += 1
            
        if cpt == MAX_SAMPLE:
            print('Max sample')
            _, max_index = max((x, (i, j))
                           for i, row in enumerate(means)
                           for j, x in enumerate(row))
            return max_index
            
        return choice_category, choice(np.flatnonzero(means[choice_category] == np.amax(means[choice_category])))

    

class MSs(MS):
    """ Murphy sampling algorithm for the group-sparse dominance """
    
    def verify_dominance(self, means):
        max_cats = [max(means[cat]) for cat in range(self.nb_categories)]
        min_cats = [min(means[cat]) for cat in range(self.nb_categories)]
        choice_category = choice(np.flatnonzero(max_cats == np.amax(max_cats)))
        del max_cats[choice_category]
        cond = (min_cats[choice_category] >= 0) and all(x <= 0 for x in max_cats)
            
        return cond, choice_category


class MS0(MS):
    """ Murphy sampling algorithm for the strong dominance """
    
    def verify_dominance(self, means):
        max_cats = [max(means[cat]) for cat in range(self.nb_categories)]
        min_cats = [min(means[cat]) for cat in range(self.nb_categories)]
        choice_category = choice(np.flatnonzero(max_cats == np.amax(max_cats)))
        del max_cats[choice_category]
        cond = all (x <= min_cats[choice_category] for x in max_cats)
            
        return cond, choice_category


class MS1(MS):
    """ Murphy sampling algorithm for the first-order dominance """
    def verify_dominance(self, means):
        # Order means
        means_ordered = np.copy(means)
        for category in range(self.nb_categories):
            means_ordered[category][::-1].sort()

        # Check cond
        temp = [means_ordered[cat][0] for cat in range(self.nb_categories)]
        choice_category = choice(np.flatnonzero(temp == np.amax(temp)))

        # Assume same number of arms
        cond = True
        ind = list(range(self.nb_categories))
        del ind[choice_category]
        for arm in range(1, self.nb_arms[0]):
            if means_ordered[choice_category][arm] < max([means_ordered[cat][arm] for cat in ind]):
                cond = False

        return cond, choice_category
   

class CatSE:
    """ Generic CatSE algorithm """
    
    def __init__(self, nb_categories, nb_arms, c=2):
        self.nb_categories = nb_categories
        if not isinstance(nb_arms, list):
            self.nb_arms = [nb_arms] * nb_categories
        else:
            self.nb_arms = nb_arms
        self.c = c

        self.t = 1
        self.cum_reward = [[0. for _ in range(self.nb_arms[i])] for i in range(nb_categories)]
        self.nb_draws = [[0. for _ in range(self.nb_arms[i])] for i in range(nb_categories)]
        
    def start_game(self):
        self.t = 1
        self.cum_reward = [[0. for _ in range(self.nb_arms[i])] for i in range(self.nb_categories)]
        self.nb_draws = [[0. for _ in range(self.nb_arms[i])] for i in range(self.nb_categories)]
    
    def get_reward(self, category, arm, reward):
        self.cum_reward[category][arm] += reward
        self.nb_draws[category][arm] += 1
        self.t += 1
    
    def play_ucb(self, category):
        index = np.array(self.cum_reward[category]) / np.array(self.nb_draws[category]) + \
                         np.sqrt(self.c * np.log(self.t) / np.array(self.nb_draws[category]))
        return category, choice(np.flatnonzero(index == np.amax(index)))

    
class CatSEs(CatSE):
    """ CatSE algorithm for group-sparse dominance """
    
    def __init__(self, nb_categories, nb_arms, c=4., threshold=0.):
        super().__init__(nb_categories, nb_arms, c)
        self.threshold = threshold
        
        self.next_action = False

    def start_game(self):
        super().start_game()
        self.next_action = False

    def choice(self):
        # Draw each arm at least once
        if self.t <= self.nb_categories * self.nb_arms[0]:
            return (self.t - 1) // self.nb_arms[0], (self.t - 1) % self.nb_arms[0]

        # Compute means
        means = [[0. for _ in range(self.nb_arms[i])] for i in range(self.nb_categories)]
        for category in range(self.nb_categories):
            for arm in range(self.nb_arms[category]):
                means[category][arm] = self.cum_reward[category][arm] / self.nb_draws[category][arm]

        # Compute set of active categories
        active_categories = []
        for category in range(self.nb_categories):
            for arm in range(self.nb_arms[category]):
                if means[category][arm] >= np.sqrt(self.c * np.log(self.nb_draws[category][arm]) /
                                                   self.nb_draws[category][arm]):
                    active_categories.append(category)
                    break

        if len(active_categories) < 1:
            # No active cateogory
            p = 1 / (np.sqrt(self.c * np.log(np.ravel(self.nb_draws)) / np.ravel(self.nb_draws)) - np.ravel(means))**2
            p = p / sum(p)

            arm = choice(self.nb_categories * self.nb_arms[0], p=p)
            return arm // self.nb_arms[0], arm % self.nb_arms[0]
        elif len(active_categories) == 1:
            # One active category
            return self.play_ucb(active_categories[0])
        else:
            # More than one active category
            active_arms = []
            for category in range(self.nb_categories):
                for arm in range(self.nb_arms[category]):
                    if means[category][arm] >= np.sqrt(self.c * np.log(self.nb_draws[category][arm]) /
                                                       self.nb_draws[category][arm]):
                        active_arms.append((category, arm))

            temp = choice(len(active_arms))
            return active_arms[temp]

        
class CatSE0(CatSE):
    """ CatSE algorithm for strong dominance """
    
    def __init__(self, nb_categories, nb_arms, c=2):
        super().__init__(nb_categories, nb_arms, c)
        
        self.next = []
        for category in range(self.nb_categories):
            for arm in range(self.nb_arms[category]):
                self.next.append((category, arm))
        self.optimal_cat = None

    def start_game(self):
        super().start_game()
        
        self.next = []
        for category in range(self.nb_categories):
            for arm in range(self.nb_arms[category]):
                self.next.append((category, arm))
        self.optimal_cat = None
    
    def choice(self):
        if len(self.next) > 0:
            return self.next.pop(0)
        elif self.optimal_cat is not None:
            return self.play_ucb(self.optimal_cat)
        else:
            self.next_move()

            if len(self.next) > 0:
                return self.next.pop(0)
            elif self.optimal_cat is not None:
                return self.play_ucb(self.optimal_cat)
            else:
                for category in range(self.nb_categories):
                    for arm in range(self.nb_arms[category]):
                        self.next.append((category, arm))
                return self.next.pop(0)
            
    def lcb(self, x, mu, error):
        return error * norm(x) - np.dot(x, mu)

    def ucb(self, x, mu, error):
        return np.dot(x, mu) + error * norm(x)

    def next_move(self):
        # Compute empirical means
        means = [[0. for _ in range(self.nb_arms[i])] for i in range(self.nb_categories)]
        for category in range(self.nb_categories):
            for arm in range(self.nb_arms[category]):
                means[category][arm] = self.cum_reward[category][arm] / self.nb_draws[category][arm]

        # Compute indices
        ucbs = np.zeros(self.nb_categories)
        lcbs = np.zeros(self.nb_categories)
        for category in range(self.nb_categories):
            x0 = np.array([1 / self.nb_arms[category]] * self.nb_arms[category])
            cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
            bounds = tuple([(0, None)] * self.nb_arms[category])
            error = np.sqrt(2 / self.nb_draws[category][0] *
                            (self.nb_arms[category] * np.log(2) + np.log(self.nb_categories * self.t)))

            res_lcb = minimize(self.lcb, x0, args=(means[category], error), bounds=bounds, constraints=cons)
            res_ucb = minimize(self.ucb, x0, args=(means[category], error), bounds=bounds, constraints=cons)
            ucbs[category] = res_ucb.fun
            lcbs[category] = - res_lcb.fun

        # Compute set active categories
        A = []
        for category in range(self.nb_categories):
            index = list(range(self.nb_categories))
            del index[category]
            if max([lcbs[cat] for cat in index]) <= ucbs[category]:
                A.append(category)

        if len(A) == 1:
            self.optimal_cat = A[0]
        elif len(A) > 1:
            for cat in A:
                for arm in range(self.nb_arms[cat]):
                    self.next.append((cat, arm))
                    
                    
class CatSE1(CatSE0):
    """ CatSE algorithm for first-order dominance """
    
    def gap_lcb(self, x, mu1, mu2, error):
        return error * norm(x) - np.dot(x, mu1 - mu2)

    def next_move(self):
        means = [[0. for _ in range(self.nb_arms[i])] for i in range(self.nb_categories)]
        for category in range(self.nb_categories):
            for arm in range(self.nb_arms[category]):
                means[category][arm] = self.cum_reward[category][arm] / self.nb_draws[category][arm]

        means_ordered = np.copy(means)
        for category in range(self.nb_categories):
            means_ordered[category][::-1].sort()

        A = list(range(self.nb_categories))
        for cat1 in range(self.nb_categories):
            if cat1 not in A:
                continue
            for cat2 in range(cat1 + 1, self.nb_categories):
                if cat2 not in A:
                    continue
                
                x0 = np.array([1 / self.nb_arms[0]] * self.nb_arms[0])
                cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
                bounds = tuple([(0, None)] * self.nb_arms[0])
                error = np.sqrt(2 / self.nb_draws[0][0] * (self.nb_arms[0] * np.log(2) + np.log(self.nb_categories * self.t)))
                res0 = minimize(self.gap_lcb, x0, args=(means_ordered[cat1], means_ordered[cat2], error),
                                bounds=bounds, constraints=cons)
                res1 = minimize(self.gap_lcb, x0, args=(means_ordered[cat2], means_ordered[cat1], error),
                                bounds=bounds, constraints=cons)

                if res0.fun <= 0:
                    A.remove(cat2)
                elif res1.fun <= 0:
                    A.remove(cat1)

        if len(A) == 1:
            self.optimal_cat = A[0]
        elif len(A) > 1:
            for cat in A:
                for arm in range(self.nb_arms[cat]):
                    self.next.append((cat, arm))
