import torch
import random
import numpy as np
from config import Config


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15,
                 sigma=0.2):
        """Initialize parameters and noise process."""
        self.config = Config.getInstance()
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        random.seed(self.config.seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()
