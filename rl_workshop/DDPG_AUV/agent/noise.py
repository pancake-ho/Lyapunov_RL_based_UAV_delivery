import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(self, actor):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.actor = actor
        self.x_prev = np.zeros_like(self.actor)
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.actor - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.actor.shape)
        self.x_prev = x
        return x