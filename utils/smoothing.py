import numpy as np

class EMASmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev = None

    def smooth(self, point):
        point = np.array(point)
        if self.prev is None:
            self.prev = point
            return point
        self.prev = self.alpha * point + (1 - self.alpha) * self.prev
        return self.prev
