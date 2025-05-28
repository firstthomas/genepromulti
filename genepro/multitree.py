from __future__ import annotations
from genepro.fos import FOS
import numpy as np
import torch.nn as nn
import torch
from genepro.node_impl import *

class Multitree(nn.Module):
  def __init__(self, n_trees: int):
    super(Multitree, self).__init__()
    self.n_trees = n_trees
    self.children = []

  def get_output_pt(self, x):
    output = []
    for child in self.children:
      output.append(child.get_output_pt(x).view(-1,1))

    return torch.cat(output,dim=1)

  def get_subtrees_consts(self):
    constants = []
    for child in self.children:
      constants.extend([node.pt_value for node in child.get_subtree() if isinstance(node, Constant)])
    return constants

  def __len__(self) -> int:
    """
    Returns the max length of the trees in the multi-tree
    """
    lens = [len(child) for child in self.children]
    return np.max(lens)

  def get_readable_repr(self) -> str:
    return [child.get_readable_repr() for child in self.children]

  def to_genotype(self, node_types: list) -> list:
    """
    Converts the multitree to a genotype representation.
    Parameters
    ----------
    node_types : list
        A list of indices representing the types of nodes in the genotype.
    Returns a list of nodes representing the genotype of the multitree.
    """
    # TODO
    return []
  
  def generate_fos(self) -> FOS:
    """
    Generates a Family of Subsets (FOS) from the multitree.
    
    Returns
    -------
    FOS
        A Family of Subsets representing the multitree.
    """
    # TODO
    return FOS([])

def generate_multitree_from_genotype(genotype: list, n_trees: int, node_types: list) -> Multitree:
    """
    Generates a multitree from a genotype.
    
    Parameters
    ----------
    genotype : list
        A list of nodes representing the genotype.
    n_trees : int
        The number of trees in the multitree.
    node_types : list
        A list of node types corresponding to the genotype.

    Returns
    -------
    Multitree
        The generated multitree.
    """
    # TODO
    return Multitree(0)