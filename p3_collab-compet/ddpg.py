import torch
from torch.optim import Adam

from config import Config
from model import Actor, Critic
from utilities import hard_update

# add OU noise for exploration
from OUNoise import OUNoise


class DDPGAgent:
    """Interacts with and learns from the environment using DDPG method."""

    def __init__(self):
        """Initialize an DDPG Agent object."""
        super(DDPGAgent, self).__init__()
        self.config = Config.getInstance()
        self.actor = Actor(self.config.state_size, self.config.action_size,
                           self.config.seed).to(self.config.device)
        self.critic = Critic(self.config.num_agents * self.config.state_size,
                             self.config.num_agents * self.config.action_size,
                             self.config.seed).to(self.config.device)
        self.target_actor = Actor(
            self.config.state_size, self.config.action_size,
            self.config.seed).to(self.config.device)
        self.target_critic = Critic(self.config.num_agents * self.config.state_size,
                                    self.config.num_agents * self.config.action_size,
                                    self.config.seed).to(self.config.device)

        self.noise = OUNoise(self.config.action_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=self.config.lr_critic,
            weight_decay=self.config.weight_decay)

    def act(self, obs, noise_decay_parameter=0.0):
        """
        Returns actions for given state as per current policy for an agent.
        """
        obs = torch.from_numpy(obs).float().to(self.config.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy()
        self.actor.train()
        action += noise_decay_parameter * self.noise.sample()
        return action

    def target_act(self, obs, noise_decay_parameter=0.0):
        """
        Returns target network actions from an agent
        """
        obs = obs.to(self.config.device)
        action = self.target_actor(
            obs) + noise_decay_parameter * self.noise.sample()
        return action

    def reset(self):
        """Reset the internal state of noise mean(mu)"""
        self.noise.reset()
