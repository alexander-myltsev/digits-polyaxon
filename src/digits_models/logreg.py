import torch
from torch.nn import Module

__all__ = [
  "Model"
]


class Model(Module):
  def __init__(self, input_shape):
    super(Model, self).__init__()
    self._model = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], 10)
    )

  def forward(self, x):
    return self._model(x)
