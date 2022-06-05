import numpy as np
from bandit_father import BanditFather


class UCB(BanditFather):
    def __init__(self, data):
        super(UCB, self).__init__(data)

    def excute(self):
        for i in range(len(self.arms)):
            if self.counts[i] == 0:
                return i
        max_ind = 0
        max_ucb = -1
        for i in range(len(self.arms)):
            v = self.values[i] + np.sqrt(
                2 * np.log(sum(self.counts)) / self.counts[i])
            if v > max_ucb:
                max_ucb = v
                max_ind = i
        return max_ind

    def update(self, reward, arm_idx):
        self.counts[arm_idx] += 1
        value = self.values[arm_idx] * (self.counts[arm_idx] - 1) + reward
        self.values[arm_idx] = value / self.counts[arm_idx]
