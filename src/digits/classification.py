import numpy as np
import torch

class DigitClassifier(object):
  def __init__(self, classifier, device, lr=1e-3):
    self.classifier = classifier
    self.device = device
    self.loss = torch.nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(
      params=self.classifier.parameters(),
      lr=lr
    )

  def fit(self, dataloader, n_epoches=32, progress=None):
    if progress is None:
      def progress(x):
        return x

    n_batches = len(dataloader)

    losses = np.ndarray(shape=(n_epoches, n_batches))

    for i in progress(range(n_epoches)):
      for j, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        self.optimizer.zero_grad()
        logits = self.classifier(X_batch)
        loss = self.loss(logits, y_batch)
        loss.backward()
        self.optimizer.step()

        losses[i, j] = loss.item()

    return losses

  def predict(self, dataloader):
    with torch.no_grad():
      predictions = []
      ground_truth = []
      for j, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(self.device)
        predictions.append(self.classifier(X_batch))
        ground_truth.append(y_batch)
      return torch.cat(predictions, dim=0).cpu().numpy(), torch.cat(ground_truth, dim=0).cpu().numpy()
