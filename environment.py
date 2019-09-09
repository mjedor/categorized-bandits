import numpy as np
from copy import deepcopy


class Result:
    """ The Result class for analyzing the output of bandit experiments. """

    def __init__(self, nb_arms, horizon):
        self.nb_arms = nb_arms
        self.choices = np.zeros(horizon, dtype=np.int)
        self.rewards = np.zeros(horizon)

    def store(self, t, choice, reward):
        self.choices[t] = choice
        self.rewards[t] = reward

    def get_nb_pulls(self):
        nb_pulls = np.zeros(self.nb_arms)
        for choice in self.choices:
            nb_pulls[choice] += 1
        return nb_pulls

    def get_regret(self, best_expectation):
        return np.cumsum(best_expectation - self.rewards)


class MAB:
    """ Multi-armed bandit problem with arms given in the 'arms' list """
    
    def __init__(self, arms):
        self.arms = arms
        self.nb_arms = len(arms)

    def play(self, policy, horizon):
        policy.start_game()
        result = Result(self.nb_arms, horizon)
        for t in range(horizon):
            choice = policy.choice()
            reward = self.arms[choice].draw()
            policy.get_reward(choice, reward)
            result.store(t, choice, reward)
        return result


class HResult:
    """ The Result class for analyzing the output of experiments of bandit with categories. """

    def __init__(self, nb_categories, nb_arms, horizon):
        self.nb_categories = nb_categories
        self.nb_arms = nb_arms
        self.choices = np.zeros((2, horizon), dtype=np.int)
        self.rewards = np.zeros(horizon)

    def store(self, t, choice_category, choice_arm, reward):
        self.choices[0, t] = choice_category
        self.choices[1, t] = choice_arm
        self.rewards[t] = reward

    def get_nb_pulls(self):
        # Works with unique nb_arms
        nb_pulls = np.zeros((self.nb_categories, self.nb_arms))
        for t in range(np.shape(self.choices)[1]):
            nb_pulls[self.choices[0, t], self.choices[1, t]] += 1
        return nb_pulls

    def get_regret(self, best_expectation):
        return np.cumsum(best_expectation - self.rewards)


class HMAB:
    """ Hierarchical multi-armed bandit problem with arms given in the 'arms' list """

    def __init__(self, arms):
        self.arms = arms
        self.nb_categories, self.nb_arms = np.shape(arms)

    def play(self, policy, horizon):
        policy_category = policy[0]
        policy_category.start_game()
        policy_arms = []
        for i in range(self.nb_categories):
            policy_arms.append(deepcopy(policy[1]))
            policy_arms[i].start_game()

        result = HResult(self.nb_categories, self.nb_arms, horizon)

        for t in range(horizon):
            # Choose category
            choice_category = policy_category.choice()
            # Choose arm inside category
            choice_arm = policy_arms[choice_category].choice()

            reward = self.arms[choice_category][choice_arm].draw()
            policy_category.get_reward(choice_category, reward)
            policy_arms[choice_category].get_reward(choice_arm, reward)
            result.store(t, choice_category, choice_arm, reward)

        return result


class FlatHMAB:
    """ Categorized multi-armed bandit problem with arms given in the 'arms' list """

    def __init__(self, arms):
        self.arms = arms
        self.nb_categories, self.nb_arms = np.shape(arms)

    def play(self, policy, horizon):
        policy.start_game()

        result = HResult(self.nb_categories, self.nb_arms, horizon)

        for t in range(horizon):
            choice_category, choice_arm = policy.choice()
            reward = self.arms[choice_category][choice_arm].draw()
            policy.get_reward(choice_category, choice_arm, reward)
            result.store(t, choice_category, choice_arm, reward)

        return result
