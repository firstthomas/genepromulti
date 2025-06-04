import numpy as np
from genepro.node import Node
import torch
import torch.nn as nn


class BooleanIf(Node):
    def __init__(self):
        super(BooleanIf, self).__init__()
        self.arity = 3
        self.symb = "Bif"
        self.type = "float"
        
        self.boolean_node_types = {GreaterThan, Equal, NotEqual}#, And, Or, Not}  # Add your boolean node types
    
    def can_have_child(self, child_node, position):
        """Only allow boolean nodes as the first child (condition)"""
        if position == 0:  # First position is the condition
            return type(child_node) in self.boolean_node_types
        return True  # Allow any node type for then/else branches
    
    def _get_args_repr(self, args):
        return f"(Bif {args[0]} then {args[1]} else {args[2]})"
    
    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # First child should already be boolean (0.0 or 1.0)
        cond = c_outs[0] > 0.5  # Use 0.5 as threshold for safety
        return np.where(cond, c_outs[1], c_outs[2])
    
    def get_output_pt(self, X):
        c_outs = self._get_child_outputs_pt(X)
        cond = c_outs[0] > 0.5
        return torch.where(cond, c_outs[1], c_outs[2])

class NotEqual(Node):
    def __init__(self):
        super(NotEqual, self).__init__()
        self.arity = 2
        self.symb = "!="
        self.type = "bool"

    def _get_args_repr(self, args):
        return f"({args[0]} != {args[1]})"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return (~np.isclose(c_outs[0], c_outs[1])).astype(float)

    def get_output_pt(self, X):
        c_outs = self._get_child_outputs_pt(X)
        return torch.where((c_outs[0] != c_outs[1]).float() > 0, torch.tensor(1.0), torch.tensor(0.0))


class Equal(Node):
    def __init__(self):
        super(Equal, self).__init__()
        self.arity = 2
        self.symb = "=="
        self.type = "bool"

    def _get_args_repr(self, args):
        return f"({args[0]} == {args[1]})"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.isclose(c_outs[0], c_outs[1]).astype(float)

    def get_output_pt(self, X):
        c_outs = self._get_child_outputs_pt(X)
        return torch.where((c_outs[0] == c_outs[1]).float() > 0, torch.tensor(1.0), torch.tensor(0.0))


class GreaterThan(Node):
    def __init__(self):
        super(GreaterThan, self).__init__()
        self.arity = 2
        self.symb = ">"
        self.type = "bool"

    def _get_args_repr(self, args):
        return f"({args[0]} > {args[1]})"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return (c_outs[0] > c_outs[1]).astype(float)

    def get_output_pt(self, X):
        c_outs = self._get_child_outputs_pt(X)
        return (c_outs[0].float() > c_outs[1].float()).float()
      
class Abs(Node):
    def __init__(self):
        super(Abs, self).__init__()
        self.arity = 1
        self.symb = "abs"
        self.type = "float"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "before")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.abs(c_outs[0])

    def get_output_pt(self, X):
        c_outs = self._get_child_outputs_pt(X)
        return torch.abs(c_outs[0]).float()
