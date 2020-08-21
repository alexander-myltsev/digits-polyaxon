import torch
from torch.nn import Module

__all__ = [
  "Model"
]


class Model(Module):
  def __init__(self, input_shape, n=8):
    super(Model, self).__init__()
    self._model = torch.nn.Sequential(
      torch.nn.Conv2d(input_shape[0], 2 * n, kernel_size=(5, 5)),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(2 * n, 3 * n, kernel_size=(3, 3)),
      torch.nn.LeakyReLU(),
      torch.nn.MaxPool2d(2),

      torch.nn.Conv2d(3 * n, 4 * n, kernel_size=(3, 3)),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(4 * n, 6 * n, kernel_size=(3, 3)),
      torch.nn.LeakyReLU(),
      torch.nn.MaxPool2d(2),

      torch.nn.Conv2d(6 * n, 8 * n, kernel_size=(3, 3)),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(8 * n, 12 * n, kernel_size=(2, 2)),
      torch.nn.LeakyReLU(),

      torch.nn.Flatten(),
      torch.nn.Linear(12 * n, 10)
    )

  def forward(self, x):
    return self._model(x)
