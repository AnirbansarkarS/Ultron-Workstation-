import numpy as np
import time

class OneEuroFilter:
    """
    Adaptive low-pass filter for noisy signals.
    Minimizes jitter at low speeds and lag at high speeds.
    """
    def __init__(self, t0=None, x0=None, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = x0
        self.dx_prev = None
        self.t_prev = t0

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def smooth(self, x, t=None):
        if t is None:
            t = time.time()
            
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = np.array(x)
            self.dx_prev = np.zeros_like(self.x_prev)
            return self.x_prev

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        x = np.array(x)
        
        # Filter derivative
        dx = (x - self.x_prev) / dt
        edx = self.dx_prev + self.alpha(self.d_cutoff, dt) * (dx - self.dx_prev)
        self.dx_prev = edx
        
        # Filter value
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        ex = self.x_prev + self.alpha(cutoff, dt) * (x - self.x_prev)
        
        self.x_prev = ex
        self.t_prev = t
        
        return ex
