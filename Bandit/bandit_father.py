from abc import ABCMeta, abstractmethod


class BanditFather(metaclass=ABCMeta):
    def __init__(self, data):
        """
        Args:
            data: array, the collection of arm, eg items
        """
        self.arms = data
        # store the number of visits per arm
        self.counts = [0] * len(self.arms)
        # store the average reward or wins of each arm
        self.values = [0.] * len(self.arms)

    @abstractmethod
    def excute(self):
        pass

    @abstractmethod
    def update(self, reward, arm_idx):
        pass
