
from genepro.multitree import Multitree
from genepro.node import Node


def subtree_crossover_fos(multitree : Multitree, multidonor : Multitree, unif_depth : int=True) -> Multitree:
  """
  Performs subtree crossover and returns the resulting offspring

  Parameters
  ----------
  tree : Node
    the tree that participates and is modified by crossover
  donor : Node
    the second tree that participates in crossover, it provides candidate subtrees
  unif_depth : bool, optional
    whether uniform random depth sampling is used to pick the root of the subtrees to swap (default is True)

  Returns
  -------
  Node
    the tree after crossover (warning: replace the original tree with the returned one to avoid undefined behavior)
  """
  return multitree

def node_level_crossover_fos(multitree : Multitree, multidonor : Multitree, same_depth : bool=False, prob_swap : float=0.1) -> Node:
  """
  Performs crossover at the level of single nodes

  Parameters
  ----------
  tree : Node
    the tree that participates and is modified by crossover
  donor : Node
    the second tree for crossover, which provides candidate nodes
  same_depth : bool, optional
    whether node-level swaps should occur only between nodes at the same depth level (default is False)
  prob_swap : float, optional
    the probability of swapping a node in tree with one in donor (default is 0.1)

  Returns
  -------
  Node
    the tree after crossover 
  """
  return multitree
