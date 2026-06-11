"""Utility functions"""

import torch, math

def xavier_uniform(shape: tuple, in_features: int, out_features: int):
  limit = math.sqrt(6.0/(in_features + out_features))
  return -2*limit*torch.rand(shape, requires_grad=True)+limit