import digits_data
import digits_models

__all__ = [
  'models',
  'datasets',
  'experiments'
]

models = {
  attr : getattr(digits_models, attr)
  for attr in digits_models.__all__
}

datasets = {
  attr : getattr(digits_data, attr)
  for attr in digits_data.__all__
}

### not all models can apply to any dataset
### in our case, however, all models can be used with all datasets
experiments = {
  '{}-{}'.format(dataset, model) : (datasets[dataset], models[model])
  for dataset in datasets
  for model in models
}