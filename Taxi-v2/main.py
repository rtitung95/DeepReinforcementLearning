from agent import Agent
from monitor import interact
import gym

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
