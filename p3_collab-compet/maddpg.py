import torch
import numpy as np
import torch.nn.functional as F

from ddpg import DDPGAgent
from config import Config
from buffer import ReplayBuffer
from utilities import soft_update


class MADDPGAgent:
    """Interacts and learns from the environment using multiple DDPG agents"""

    def __init__(self):
        """Initialize a MADDPG Agent object."""
        super(MADDPGAgent, self).__init__()
        self.config = Config.getInstance()
        self.action_num = self.config.action_size * self.config.num_agents
        self.t_step = 0

        self.maddpg_agent = [DDPGAgent()
                             for _ in range(self.config.num_agents)]

        self.memory = ReplayBuffer()

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    # def get_target_actors(self):
    #     """get target_actors of all the agents in the MADDPG object"""
    #     target_actors = [
    #         ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
    #     return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(
            self.maddpg_agent, obs_all_agents)]
        return np.concatenate(actions)

    def update_act(self, obs_all_agents, agent_num, noise_decay_parameter=0.0):
        """
        get target network actions from all the agents in the MADDPG object
        """
        actions_ = []
        for a_i, ddpg_agent in enumerate(self.maddpg_agent):
            obs = obs_all_agents[:, a_i, :].to(self.config.device)
            acn = ddpg_agent.actor(
                obs) + noise_decay_parameter * ddpg_agent.noise.sample()
            if a_i != agent_num:
                acn = acn.detach()
            actions_.append(acn)
        return actions_

    def target_act(self, obs_all_agents, noise=0.0):
        """
        get target network actions from all the agents in the MADDPG object
        """
        target_actions = [ddpg_agent.target_act(
            obs_all_agents[:, a_i, :], noise) for a_i, ddpg_agent in enumerate(self.maddpg_agent)]
        return target_actions

    def step(self, _states, _actions, _rewards, _next_states, _dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        states_full = np.reshape(_states, newshape=(-1))
        next_states_full = np.reshape(_next_states, newshape=(-1))
        self.memory.add(_states, states_full, _actions, _rewards, _next_states,
                        next_states_full,  _dones)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.update_every

        if self.t_step == 0:
            if len(self.memory) > self.config.batch_size:
                for a_i in range(self.config.num_agents):
                    samples = self.memory.sample()
                    self.update(samples, a_i)
                self.update_targets()

    def update_critic(self, samples, agent_number):
        """Update critic weights"""
        states, states_full, actions, rewards, next_states, next_states_full, dones = samples
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        # ---------------------------- update critic ---------------------- #
        actions_next = self.target_act(next_states)
        actions_next = torch.cat(actions_next, dim=1)

        Q_target_next = agent.target_critic(next_states_full, actions_next)
        Q_targets = rewards[:, agent_number].view(-1, 1) + self.config.gamma * \
            Q_target_next * (1 - dones[:, agent_number].view(-1, 1))
        Q_expected = agent.critic(
            states_full, actions.reshape(-1, self.action_num))
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        agent.critic_optimizer.step()

    def update_actor(self, samples, agent_number):
        """Update actor weights"""
        states, states_full, actions, rewards, next_states, next_states_full, dones = samples
        agent = self.maddpg_agent[agent_number]

        agent.actor_optimizer.zero_grad()
        actions_pred = self.update_act(states, agent_number)
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -agent.critic(states_full, actions_pred).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        # ---------------------------- update critic ---------------------- #
        self.update_critic(samples, agent_number)

        # ---------------------------- update actor ------------------------- #
        self.update_actor(samples, agent_number)

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor,
                        ddpg_agent.actor, self.config.tau)
            soft_update(ddpg_agent.target_critic,
                        ddpg_agent.critic, self.config.tau)

    def reset(self):
        """Resets weight of all agents"""
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()
