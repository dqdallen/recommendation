import numpy as np


class DisjointLibUCB:
    def __init__(self, data, dim, alpha=0.25):
        self.arms = data
        self.dim = dim
        self.alpha = alpha
        self.A = []
        self.invA = []
        self.b = []
        self.theta = []
        self.initialize()

    def initialize(self):
        for i in range(len(self.arms)):
            self.A.append(np.eye(self.dim))
            self.invA.append(np.eye(self.dim))
            self.b.append(np.zeros((self.dim, 1)))

    def excute(self, features):
        # feature: d * 1
        max_prob = -1
        max_ind = 0
        for i in range(len(self.invA)):
            theta = np.dot(self.invA[i], self.b[i])
            score = np.dot(theta.T, features[:, i])[0]
            bound = self.alpha * np.sqrt(np.dot(
                np.dot(features[:, i].T, self.invA[i]), features[:, i]))
            prob_tmp = score + bound
            if max_prob < prob_tmp:
                max_prob = prob_tmp
                max_ind = i

        return max_ind

    def update(self, features, reward, arm_idx):
        fea_tmp = features[:, arm_idx]
        self.A[arm_idx] = self.A[arm_idx] + np.dot(fea_tmp, fea_tmp.T)
        self.b[arm_idx] = self.b[arm_idx] + reward * fea_tmp
        self.invA[arm_idx] = np.linalg.inv(self.A[arm_idx])
