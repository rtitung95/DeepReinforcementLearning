import torch
import random
import numpy as np
from collections import deque, namedtuple

from config import Config


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self):
        """Initialize a ReplayBuffer object."""
        self.config = Config.getInstance()
        self.memory = deque(maxlen=int(self.config.buffer_size))
        self.batch_size = int(self.config.batch_size)
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "state_full", "action",
                                     "reward", "next_state",
                                     "next_state_full",
                                     "done"])
        random.seed(self.config.seed)

    def add(self, state, state_full, action, reward, next_state,
            next_state_full, done):
        """Add a new experience to memory."""
        e = self.experience(state, state_full, action, reward, next_state,
                            next_state_full, done)
        self.memory.append(e)

    def convert_to_tensor(self, attributes_list):
        """Convert a list to tensor"""
        return torch.from_numpy(np.array(attributes_list)).float().to(self.config.device)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.convert_to_tensor(
            [e.state for e in experiences if e is not None])
        states_full = self.convert_to_tensor(
            [e.state_full for e in experiences if e is not None])
        actions = self.convert_to_tensor(
            [e.action for e in experiences if e is not None])
        rewards = self.convert_to_tensor(
            [e.reward for e in experiences if e is not None])
        next_states = self.convert_to_tensor(
            [e.next_state for e in experiences if e is not None])
        next_states_full = self.convert_to_tensor(
            [e.next_state_full for e in experiences if e is not None])
        dones = self.convert_to_tensor(
            np.array([e.done for e in experiences if e is not None]).astype(np.uint8))

        return (states, states_full, actions, rewards, next_states,
                next_states_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
