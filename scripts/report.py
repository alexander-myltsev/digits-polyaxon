from digits_utils import as_table, as_tabular

def report(report_dir, output_dir, quiet=False):
  import os
  import json
  from collections import defaultdict

  results = defaultdict(dict)

  for item in os.listdir(report_dir):
    try:
      if not item.endswith('.json'):
        continue
      path = os.path.join(report_dir, item)
      with open(path, 'r') as f:
        record = json.load(f)

      dataset = record['dataset']
      model = record['model']
      results[dataset][model] = dict(
        accuracy_train=record['accuracy_train'],
        accuracy_test=record['accuracy_test'],
      )
    except:
      import warnings
      import traceback
      warnings.warn(traceback.format_exc())

  for dataset in results:
    try:
      path = os.path.join(output_dir, '{dataset}.tex'.format(dataset=dataset))
      with open(path, 'w') as f:
        f.write(as_tabular(results[dataset]))
    except:
      import warnings
      import traceback
      warnings.warn(traceback.format_exc())

    try:
      path = os.path.join(output_dir, '{dataset}.txt'.format(dataset=dataset))
      table = as_table(results[dataset])
      with open(path, 'w') as f:
        f.write(table)

      if not quiet:
        print(dataset)
        print(table)
    except:
      import warnings
      import traceback
      warnings.warn(traceback.format_exc())

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(prog='report')
  parser.add_argument('report_dir', type=str, default='reports/')
  parser.add_argument('output_dir', type=str, default='reports/')
  parser.add_argument('--quiet', action='store_true', default=False)

  arguments = parser.parse_args()
  report(arguments.report_dir, arguments.output_dir, quiet=arguments.quiet)