import gymnasium as gym

from genepro.node_impl import *
from genepro.custom_node_impl import *
from genepro.evo import Evolution
from genepro.node_impl import Constant

import torch

import random
from collections import namedtuple, deque

import matplotlib.pyplot as plt
from genepro.custom_node_impl import *
from datetime import datetime
import csv

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
      
      action = torch.argmax(multitree.get_output_pt(input_sample))
      observation, reward, terminated, truncated, info = env.step(action.item())
      rewards.append(reward)
      output_sample = torch.from_numpy(observation.reshape((1,-1))).float()
      memory.push(input_sample, torch.tensor([[action.item()]]), output_sample, torch.tensor([reward]))
      if (terminated or truncated) and not ignore_done:
        break

  fitness = np.sum(rewards)
  return fitness, memory


def get_test_score(tree):
    rewards = []

    for i in range(10):
      # get initial state
      observation = env.reset(seed=i)
      observation = observation[0]

      for _ in range(500):    
        # build up the input sample for GP
        input_sample = torch.from_numpy(observation.reshape((1,-1))).float()
        # get output (squeezing because it is encapsulated in an array)
        output = tree.get_output_pt(input_sample)
        action = torch.argmax(output)# What goes here?
        observation, reward, terminated, truncated, info = env.step(action.item())
        rewards.append(reward)

        # print(output.shape)
        # print("observation:", observation)
        # print(f"Action: {action.item()}, Reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
        # build up the output sample for GP
        output_sample = torch.from_numpy(observation.reshape((1,-1))).float()
        if (terminated or truncated):
            break

    fitness = np.sum(rewards)
    
    return fitness


def test_operators(internal_nodes_sets, internal_nodes_sets_names):
    num_features = env.observation_space.shape[0]
    leaf_nodes = [Feature(i) for i in range(num_features)]
    leaf_nodes = leaf_nodes + [Constant()]
    num_of_experiments = 5

    data = []
    for (i, internal_nodes) in enumerate(internal_nodes_sets):
        total_time = 0
        total_score = 0
        for j in range(num_of_experiments):
            start = datetime.now()
            evo = Evolution(
                fitness_function=fitness_function_pt, 
                internal_nodes=internal_nodes,
                leaf_nodes=leaf_nodes,
                n_trees=4,
                pop_size=24,
                max_gens=10,
                max_tree_size=31,
                n_jobs=8,
                verbose=True)
            evo.evolve()
            end = datetime.now()
            time = (end-start).total_seconds()
            best = evo.best_of_gens[-1]
            test_score = get_test_score(best)
            total_time += time
            total_score += test_score

        avg_time = total_time / num_of_experiments
        avg_score = total_score / num_of_experiments
        avg_score_per_second = avg_score / avg_time
        print(f"{internal_nodes_sets_names[i]} operators, avg_time {avg_time}, avg_score {avg_score}, avg_score_per_second {avg_score_per_second}")
        data.append([internal_nodes_sets_names[i], avg_time, avg_score, avg_score_per_second])

    try:
        with open("results_operators/test_operators.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Operator Set", "Avg Time (seconds)", "Avg Test Score", "Avg Score per Second"])
            writer.writerows(data)
        print("CSV file written successfully.")
    except Exception as e:
        print("Error while writing CSV file:", e)
    return data

def plot(data):
    operator_sets = [entry[0] for entry in data]
    avg_times = [entry[1] for entry in data]
    avg_scores = [entry[2] for entry in data]
    avg_scores_per_second = [entry[3] for entry in data]

    # Plot Avg Score per Second vs Operator Set
    plt.figure(figsize=(10, 6))
    plt.plot(operator_sets, avg_scores_per_second, marker='o', label='Avg Score per Second', color='blue')
    plt.title('Avg Score per Second vs Operator Set')
    plt.xlabel('Operator Set')
    plt.ylabel('Avg Score per Second')
    plt.grid(True)
    plt.legend()
    plt.savefig("results_operators/avg_score_per_second_vs_operator_set.png")

    # Plot Avg Time vs Operator Set
    plt.figure(figsize=(10, 6))
    plt.plot(operator_sets, avg_times, marker='o', label='Avg Time (seconds)', color='green')
    plt.title('Avg Time vs Operator Set')
    plt.xlabel('Operator Set')
    plt.ylabel('Avg Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig("results_operators/avg_time_vs_operator_set.png")

    # Plot Avg Score vs Operator Set
    plt.figure(figsize=(10, 6))
    plt.plot(operator_sets, avg_scores, marker='o', label='Avg Score', color='red')
    plt.title('Avg Score vs Operator Set')
    plt.xlabel('Operator Set')
    plt.ylabel('Avg Score')
    plt.grid(True)
    plt.legend()
    plt.savefig("results_operators/avg_score_vs_operator_set.png")

def load_csv(filename):
    data = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append([int(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return data

    
internal_nodes_baseline = [Plus(),Minus(),Times(),Div()]
internal_nodes_math = [Plus(),Minus(),Times(),Div(),Min(),Max(),Abs()]
internal_nodes_bool = [Plus(),Minus(),Times(),Div(),BooleanIf(),GreaterThan()]
internal_nodes_transcendental = [Plus(),Minus(),Times(),Div(),Sin(),Cos(),Exp(),Log()]
internal_nodes_all = [Plus(),Minus(),Times(),Div(),Min(),Max(),Abs(),BooleanIf(),GreaterThan(),Sin(),Cos(),Exp(),Log()]
internal_nodes_sets = [internal_nodes_baseline, internal_nodes_math, internal_nodes_bool, internal_nodes_transcendental, internal_nodes_all]
internal_nodes_sets_names = ["Baseline", "Math", "Boolean", "Transcendental", "All"]

data = test_operators(internal_nodes_sets, internal_nodes_sets_names)
# data = load_csv("results_operators/test_operators.csv")
plot(data)