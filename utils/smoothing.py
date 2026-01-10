import numpy as np

class EMASmoother:
    def __init__(self, base_alpha=0.85, boost_alpha=0.95):
        self.base_alpha = base_alpha
        self.boost_alpha = boost_alpha
        self.prev = None

    def smooth(self, value):
        value = np.array(value)

        if self.prev is None:
            self.prev = value
            return value

        velocity = np.linalg.norm(value - self.prev)

        # Faster movement → less smoothing
        alpha = self.boost_alpha if velocity > 0.01 else self.base_alpha

        self.prev = alpha * value + (1 - alpha) * self.prev
        return self.prev
