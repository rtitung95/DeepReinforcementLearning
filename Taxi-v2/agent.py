import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = 0.005
        if np.random.random() > epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        gamma = 1.0
        alpha = 0.01
        epsilon = 0.005
        current_q = self.Q[state][action]
        policy_s = (np.ones(self.nA) * epsilon)/self.nA
        max_val_state = np.argmax(self.Q[next_state])
        policy_s[max_val_state] += 1 - epsilon
        q_next = np.dot(self.Q[next_state], policy_s)
        target = reward + gamma * q_next
        current_q += alpha * (target - current_q)
        self.Q[state][action] = current_q
