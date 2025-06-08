import os
import gymnasium as gym

from genepro.node_impl import *
from genepro.evo import Evolution
from genepro.node_impl import Constant

import torch

import random
from collections import namedtuple, deque
import numpy as np

# Test different crossover rates
from genepro.variation import *
from datetime import datetime

from genepro.selection import tournament_selection
from genepro.variation import node_level_crossover, subtree_crossover, subtree_mutation

env = gym.make("LunarLander-v3", render_mode="rgb_array")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

filename = "test_scores_subtree_crossover.csv"


# RESOURCES
POPULATION_SIZE = 104
MAX_TREE_SIZE = 31
NUM_GENERATIONS = 20
NUM_EXPERIMENTS = 1
# HYPERPARAMETERS
CROSSOVER_RATE_SUBTREE = 0.5
CROSSOVER_RATE_NODE_LEVEL = 0.7
MUTATION_RATE = 0.4
NUMBER_OF_CONSTANTS = 4
TOURNAMENT_SIZE = 8

VERBOSE = True
EXPERIMENT_CONTINUE = 0
CONTINUE_AFTER_FINISHED = False

num_features = env.observation_space.shape[0] # type: ignore
leaf_nodes = [Feature(i) for i in range(num_features)]
leaf_nodes = leaf_nodes + [Constant() for _ in range(NUMBER_OF_CONSTANTS)]
# Use all internal nodes found in node_impl.py
internal_nodes = [Plus(), Minus(), Times(), Div(), Sin(), Cos(), Log(), Min(), Max()]



def fitness_function_pt(multitree, num_episodes=5, episode_duration=300, render=False, ignore_done=False):
  memory = ReplayMemory(10000)
  rewards = []

  for _ in range(num_episodes):
    # get initial state of the environment
    observation = env.reset()
    observation = observation[0]
    
    for _ in range(episode_duration):

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


        output_sample = torch.from_numpy(observation.reshape((1,-1))).float()
        if (terminated or truncated):
            break

    fitness = np.sum(rewards)
    
    return fitness


def experiment(population_size, max_tree_size, num_generations, crossover_rate_subtree, crossover_rate_node_level, mutation_rate, tournament_size, number_of_experiments, fitness_function_pt, leaf_nodes, internal_nodes, filename):
    # Check if file exists and read existing lines
    if not os.path.exists(filename):
      with open(filename, "w+") as f:
        f.write("experiment,generation,test_score,population_size,max_tree_size,crossover_rate_subtree,crossover_rate_node_level,mutation_rate,tournament_size\n")
    with open(filename, "a", buffering=1) as f:  # append mode
      
      times_per_experiment = []
      for i in range(number_of_experiments):
        start_time = datetime.now()
        evo = evolve(population_size, max_tree_size, num_generations, crossover_rate_subtree, crossover_rate_node_level, mutation_rate, tournament_size, fitness_function_pt, leaf_nodes, internal_nodes)
        best_of_gens = evo.best_of_gens
        for gen_index, best in enumerate(best_of_gens):
          test_score = get_test_score(best)
          f.write(f"{i},{gen_index},{test_score},{population_size},{max_tree_size},{crossover_rate_subtree},{crossover_rate_node_level},{mutation_rate},{tournament_size}\n")
          f.flush()  # ensure data is written to disk immediately
        times_per_experiment.append((datetime.now() - start_time))
        print(f"Experiment {i + 1} of {number_of_experiments} completed in {times_per_experiment[-1]}. Estimated time remaining: {(number_of_experiments - (i + 1)) * np.mean(times_per_experiment)} seconds\navg time per experiment: {np.mean(times_per_experiment)} seconds")

def tune_hyperparameters(population_size, max_tree_size, num_generations, fitness_function_pt, leaf_nodes, internal_nodes, number_of_experiments, filename):
    crossover_rates_subtree = [0.5]
    crossover_rates_node_level = [0.1, 0.3, 0.5, 0.7, 0.9]
    mutation_rates = [MUTATION_RATE]
    tournament_sizes = [TOURNAMENT_SIZE]

    total_number_of_experiments = len(crossover_rates_subtree) * len(crossover_rates_node_level) * len(mutation_rates) * len(tournament_sizes) * number_of_experiments
    experiment_count = 0
    times_per_experiment = []

    # Check if file exists and read existing lines
    if not os.path.exists(filename):
      with open(filename, "w+") as f:
        f.write("crossover_rate_subtree,crossover_rate_node_level,mutation_rate,tournament_size,experiment,generation,test_score,population\n")
        
    with open(filename, "a", buffering=1) as f:  # append mode
      for tournament_size in tournament_sizes:
        for crossover_rate_subtree in crossover_rates_subtree:
            for crossover_rate_node_level in crossover_rates_node_level:
                for mutation_rate in mutation_rates:
                    for i in range(number_of_experiments):
                      experiment_count += 1
                      if experiment_count < EXPERIMENT_CONTINUE:
                        continue
                      start_time = datetime.now()
                      print(f"experiment {experiment_count} of {total_number_of_experiments} estimated time remaining: {(total_number_of_experiments - experiment_count) * (np.mean(times_per_experiment) if times_per_experiment else 0)} seconds")
                      evo = evolve(population_size, max_tree_size, num_generations, crossover_rate_subtree, crossover_rate_node_level, mutation_rate, tournament_size, fitness_function_pt, leaf_nodes, internal_nodes)
                      for gen_index, best in enumerate(evo.best_of_gens):
                        test_score = get_test_score(best)
                        f.write(f"{crossover_rate_subtree},{crossover_rate_node_level},{mutation_rate},{tournament_size},{i},{gen_index},{test_score},{population_size}\n")
                        f.flush()  # ensure data is written to disk immediately
                      times_per_experiment.append((datetime.now() - start_time))


def evolve(population_size, max_tree_size, num_generations, crossover_rate_subtree, crossover_rate_node_level, mutation_rate, tournament_size, fitness_function_pt, leaf_nodes, internal_nodes):
    evo = Evolution(
        fitness_function_pt, internal_nodes, leaf_nodes,
        4,
        pop_size=population_size,
        max_gens=num_generations,
        max_tree_size=max_tree_size,
        n_jobs=8,
        verbose=VERBOSE,
        crossovers=[{"fun": subtree_crossover, "rate": crossover_rate_subtree}],
        mutations=[{"fun": subtree_mutation, "rate": mutation_rate}],
        selection={"fun":tournament_selection,"kwargs":{"tournament_size":tournament_size}},
      )
    evo.evolve()
    return evo

if __name__ == "__main__":
  while True:
    start_time = datetime.now()
    CROSSOVER_RATE_SUBTREE = 0.5
    CROSSOVER_RATE_NODE_LEVEL = 0.7
    filename = "test_scores_node_crossover.csv"
    experiment(POPULATION_SIZE, MAX_TREE_SIZE, NUM_GENERATIONS, CROSSOVER_RATE_SUBTREE, CROSSOVER_RATE_NODE_LEVEL, MUTATION_RATE, TOURNAMENT_SIZE, NUM_EXPERIMENTS, fitness_function_pt, leaf_nodes, internal_nodes, filename)

    # tune_hyperparameters(POPULATION_SIZE, MAX_TREE_SIZE, NUM_GENERATIONS, fitness_function_pt, leaf_nodes, internal_nodes, NUM_EXPERIMENTS, "tuning_results.csv")
    print("Experiment finished in", datetime.now() - start_time)

    CROSSOVER_RATE_SUBTREE = 0.7
    CROSSOVER_RATE_NODE_LEVEL = 0
    filename = "test_scores_subtree_crossover.csv"
    experiment(POPULATION_SIZE, MAX_TREE_SIZE, NUM_GENERATIONS, CROSSOVER_RATE_SUBTREE, CROSSOVER_RATE_NODE_LEVEL, MUTATION_RATE, TOURNAMENT_SIZE, NUM_EXPERIMENTS, fitness_function_pt, leaf_nodes, internal_nodes, filename)


  