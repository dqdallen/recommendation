from egreedy import EpsilonGreedy
from lin_ucb import DisjointLibUCB
from thompson_sampling import ThompsonSampling
from ucb import UCB
import numpy as np


np.random.seed(2022)
data = np.random.randn(100, 10)
label = [0] * 50 + [1] * 50
np.random.shuffle(label)

eg = EpsilonGreedy(data, eps=0.5)
lucb = DisjointLibUCB(data, 10)
ts = ThompsonSampling(data)
ub = UCB(data)
for i in range(100):
    arm_idx = eg.excute()
    eg.update(label[arm_idx], arm_idx)
    print(arm_idx)
    arm_idx = lucb.excute(data.T)
    lucb.update(data.T, label[arm_idx], arm_idx)
    print(arm_idx)
    arm_idx = ts.excute()
    ts.update(label[arm_idx], arm_idx)
    print(arm_idx)
    arm_idx = ub.excute()
    ub.update(label[arm_idx], arm_idx)
    print(arm_idx)
