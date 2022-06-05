import numpy as np
from bandit_father import BanditFather


class EpsilonGreedy(BanditFather):
    def __init__(self, data, eps):
        """The implementation of epsilon_greedy
        Args:
            eps: the epsilon
        """
        super(EpsilonGreedy, self).__init__(data)
        self.eps = eps

    def get_best_armidx(self):
        v = max(self.values)
        return self.values.index(v)

    def get_random_armidx(self):
        idx = np.random.randint(len(self.arms))
        return idx

    def update(self, reward, arm_idx):
        self.counts[arm_idx] += 1
        new_value = self.values[arm_idx] * (self.counts[arm_idx] - 1.) + reward
        new_value = new_value / self.counts[arm_idx]
        self.values[arm_idx] = new_value

    def excute(self):
        if np.random.random() > self.eps:
            arm_idx = self.get_best_armidx()
        else:
            arm_idx = self.get_random_armidx()

        return arm_idx
