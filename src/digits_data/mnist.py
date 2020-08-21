from torchvision import datasets, transforms

__all__ = [
  "Dataset"
]


class Dataset(object):
  def __init__(self, dataroot):
    self._transform = transforms.Compose([
      transforms.Resize(size=(32, 32)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,)),
    ])
    self._train_set = datasets.MNIST(dataroot, download=True, train=True, transform=self._transform)
    self._test_set = datasets.MNIST(dataroot, download=True, train=False, transform=self._transform)

  @property
  def train_set(self):
    return self._train_set

  @property
  def test_set(self):
    return self._test_set
