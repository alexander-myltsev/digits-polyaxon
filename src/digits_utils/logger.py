import PIL
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from comet_ml import Experiment as CometExperiment
from polyaxon import tracking

__all__ = [
  'LocalLogger', 'CometLogger', 'PolyaxonLogger',
  'get_logger',
]

def warn():
  import traceback
  import warnings
  warnings.warn(traceback.format_exc())

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

class Logger(object):
  def log_metrics(self, dataset_name, model_name, **kwargs):
    raise NotImplementedError()

  def log_losses(self, dataset_name, model_name, losses):
    raise NotImplementedError()


class LocalLogger(Logger):
  """
  Writing json logger
  """
  def __init__(self, root):
    from .common import ensure_directories
    self._report_root, self._figure_root = ensure_directories(root, 'reports/', 'figures/')
    
    super(LocalLogger, self).__init__()

  def log_metrics(self, dataset_name, model_name, **info):
    path = os.path.join(
      self._report_root,
      '{dataset}-{model}.json'.format(dataset=dataset_name, model=model_name)
    )

    info['dataset'] = dataset_name
    info['model'] = model_name
    with open(path, 'w') as f:
      json.dump(info, f, indent=2)

  def _log_learning_curve(self, dataset_name, model_name, losses):
    from .viz import make_learning_curve

    f = make_learning_curve(dataset_name, model_name, losses)
    plt.savefig(
      os.path.join(
        self._figure_root,
        '{dataset}-{model}.png'.format(dataset=dataset_name, model=model_name)
      )
    )
    return f

  def log_losses(self, dataset_name, model_name, losses):
    f = self._log_learning_curve(dataset_name, model_name, losses)
    plt.close(f)


class CometLogger(LocalLogger):
  """
  Comet ml logger
  """
  def __init__(self, root, experiment : CometExperiment):
    self._experiment = experiment
    
    super(CometLogger, self).__init__(root)

  def log_metrics(self, dataset_name, model_name, **info):
    super(CometLogger, self).log_metrics(dataset_name, model_name, **info)

    for metric_name, value in info.items():
      self._experiment.log_metric(
        '{dataset}_{model}_{metric}'.format(dataset=dataset_name, model=model_name, metric=metric_name),
        value
      )

  def log_losses(self, dataset_name, model_name, losses):
    f = self._log_learning_curve(dataset_name, model_name, losses)
    self._experiment.log_figure(
      "Losses-{}".format(self._experiment.project_name),
      f
    )
    plt.close(f)


class PolyaxonLogger(LocalLogger):
  """
  Polyaxon logger
  """

  def __init__(self, root):
    tracking.init()

    super(PolyaxonLogger, self).__init__(root)

  def log_metrics(self, dataset_name, model_name, **info):
    super(PolyaxonLogger, self).log_metrics(dataset_name, model_name, **info)

    for metric_name, value in info.items():
      tracking.log_metric(
        '{dataset}_{model}_{metric}'.format(dataset=dataset_name, model=model_name, metric=metric_name),
        value,
      )

  def log_losses(self, dataset_name, model_name, losses):
    f = self._log_learning_curve(dataset_name, model_name, losses)

    # lst = list(f.canvas.get_width_height())
    # lst.append(3)
    # img = PIL.Image.fromarray(np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8).reshape(lst))

    # print(f)
    # data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
    # print(data.shape)

    # width, height = f.get_size_inches() * f.get_dpi()
    # data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    # tracking.log_image(
    #   data=data,
    #   name="Losses",
    # )
    plt.close(f)


def get_logger(logger, root, project=None, workspace=None) -> Logger:
  from digits_utils import LocalLogger, CometLogger

  if logger.lower() == "local":
    return LocalLogger(root)

  elif logger.lower() == "comet":
    assert project is not None, 'for comet logger, please, provide project name'
    assert workspace is not None, 'for comet logger, please, provide workspace'

    experiment = CometExperiment(project_name=project, workspace=workspace)
    return CometLogger(root=root, experiment=experiment)

  elif logger.lower() == "polyaxon":
    polyaxon_logger = PolyaxonLogger(root=root)
    return polyaxon_logger

  else:
    raise ValueError("Unknown experiment context")