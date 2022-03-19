import random
from typing import List


class THSpace:
    """
    Threshold Space to be tried out.
    List of X's, that'll enter the
        model (ThOpt).
    A threshold space with higher
    sensitivity will lead to a better
    optimization, as well as a longer
    time of estimation.
    """

    def __init__(self, space):
        self.space = space
        self.tried = []

    @property
    def triable(self) -> List[float]:
        return [i for i in self.space
                if i not in self.tried]

    def take(self, randomly=False):
        if s := self.triable:
            if randomly:
                th = random.choice(s)
            else:
                th = s[0]
            self.tried.append(th)
            return th
        else:
            return None
