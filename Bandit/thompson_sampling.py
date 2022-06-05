import numpy as np
from bandit_father import BanditFather


class ThompsonSampling(BanditFather):
    def __init__(self, data):
        super(ThompsonSampling, self).__init__(data)

    def update(self, reward, arm_idx):
        if reward == 1:
            self.values[arm_idx] += 1
        self.counts[arm_idx] += 1

    def excute(self):
        # sample from beta distribution
        pbeta = [np.random.beta(a + 1, b - a + 1)
                 for a, b in zip(self.values, self.counts)]
        arm_idx = np.argmax(pbeta)
        return arm_idx
