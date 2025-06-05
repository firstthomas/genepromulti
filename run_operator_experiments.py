import gymnasium as gym

from genepro.node_impl import *
from genepro.custom_node_impl import *
from genepro.evo import Evolution
from genepro.node_impl import Constant

import torch
import torch.optim as optim

import random
import os
import copy
from collections import namedtuple, deque

import matplotlib.pyplot as plt
from matplotlib import animation
from genepro.custom_node_impl import *
from genepro.variation import generate_random_tree

env = gym.make("LunarLander-v3", render_mode="rgb_array")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
      self.memory += other.memory
      return self 

    def __add__(self, other):
      self.memory = self.memory + other.memory 
      return self
    
frames = []

def fitness_function_pt(multitree, num_episodes=5, episode_duration=300, render=False, ignore_done=False):
  memory = ReplayMemory(10000)
  rewards = []

  for _ in range(num_episodes):
    # get initial state of the environment
    observation = env.reset()
    observation = observation[0]
    for _ in range(episode_duration):
      if render:
        frames.append(env.render())

      input_sample = torch.from_numpy(observation.reshape((1,-1))).float()
      
      # what goes here? TODO
      action = torch.argmax(multitree.get_output_pt(input_sample))
      observation, reward, terminated, truncated, info = env.step(action.item())
      rewards.append(reward)
      output_sample = torch.from_numpy(observation.reshape((1,-1))).float()
      memory.push(input_sample, torch.tensor([[action.item()]]), output_sample, torch.tensor([reward]))
      if (terminated or truncated) and not ignore_done:
        break

  fitness = np.sum(rewards)
  return fitness, memory

def fitness_function_pt_new(multitree, num_episodes=5, episode_duration=300, render=False, ignore_done=False):
  memory = ReplayMemory(10000)
  rewards = []
  total_reward = 0
  shaped_reward = 0
  successful_landings = 0

  for ep in range(num_episodes):
      ep_reward = 0
      ep_shaped = 0
      ep_steps = 0
      landed = False

      observation = env.reset()[0]
      for t in range(episode_duration):
          if render:
              frames.append(env.render())
          input_sample = torch.from_numpy(observation.reshape((1,-1))).float()
          action_tensor = multitree.get_output_pt(input_sample)
          action_probs = torch.softmax(action_tensor, dim=1)
          print(action_probs)

          action = torch.multinomial(action_probs, num_samples=1).item()
        #   print(f"Action: {action}")          
          observation, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
          total_reward += reward
          ep_reward += reward
          ep_steps += 1

          # Heuristic shaping
          pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1_contact, leg2_contact = observation

          # Encourage staying near the center (x=0), and low lateral/vertical velocity
          position_penalty = -10.0 * abs(pos_x)
          velocity_penalty = -5.0 * (abs(vel_x) + abs(vel_y))

          # Penalize rotation and spinning
          rotation_penalty = -20.0 * abs(angle) - 10.0 * abs(ang_vel)

          # Bonus for making contact with both legs
          leg_contact_bonus = 10.0 * (leg1_contact + leg2_contact)

          # Bonus for hovering gently close to the ground without crashing
          hover_bonus = 0
          if pos_y < 0.25 and abs(vel_y) < 0.5 and not (leg1_contact or leg2_contact):
              hover_bonus = 10.0  # gentle hover

          # Strong bonus for soft landing
          landing_bonus = 0
          if done and reward == 100:
              landing_bonus = 100.0

          # Crash penalty
          crash_penalty = -100.0 if done and reward == -100 else 0

          # Total shaped reward
          shaped = (
              position_penalty +
              velocity_penalty +
              rotation_penalty +
              leg_contact_bonus +
              hover_bonus +
              landing_bonus +
              crash_penalty
          )
          ep_shaped += shaped

      shaped_reward += ep_shaped
      if landed:
          successful_landings += 1
    
  fitness = (
    0.3 * total_reward +
    0.5 * shaped_reward / num_episodes +
    50.0 * successful_landings  
  )
  return fitness, memory


num_features = env.observation_space.shape[0]
print("Number of features:", num_features)
leaf_nodes = [Feature(i) for i in range(num_features)]
leaf_nodes = leaf_nodes + [Constant()] # Think about the probability of sampling a coefficient
internal_nodes_baseline = [Plus(),Minus(),Times(),Div()]
internal_nodes_mma = [Plus(),Minus(),Times(),Div(),Min(),Max(),Abs()]
internal_nodes_bool = [Plus(),Minus(),Times(),Div(),BooleanIf(),GreaterThan()]
internal_nodes_trans = [Plus(),Minus(),Times(),Div(),Sin(),Cos(),Exp(),Log()]
internal_nodes_all = [Plus(),Minus(),Times(),Div(),Min(),Max(),Abs(),BooleanIf(),GreaterThan(),Sin(),Cos(),Exp(),Log()]
internal_nodes = internal_nodes_baseline

evo = Evolution(
  fitness_function_pt, internal_nodes, leaf_nodes,
  4,
  pop_size=48,
  max_gens=10,
  max_tree_size=31,
  n_jobs=8,
  verbose=True)


bool_nodes = [node for node in internal_nodes if node.type == "bool"]
float_nodes = [node for node in internal_nodes if node.type == "float"]
trees = [generate_random_tree(
    bool_nodes=bool_nodes,
    float_nodes=float_nodes,
    leaf_nodes=leaf_nodes,
    max_depth=6,
    curr_depth=0
) for _ in range(5)]
for t in trees:
    print(t.get_readable_repr())
    print()

evo.evolve()