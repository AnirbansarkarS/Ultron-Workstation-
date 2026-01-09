"""
Smoothing, Kalman, EMA filters for landmark data.
"""

class EMAFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.state = None

    def apply(self, value):
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state
