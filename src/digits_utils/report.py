__all__ = [
  'as_tabular', 'as_table'
]

def as_tabular(results):
  import numpy as np

  best_indx = np.argmax([results[model_name]['accuracy_test'] for model_name in results])

  hline = '\\hline'

  rows = [
    '  {hline}\n  model & accuracy test & accuracy train\\\\\n  {hline}'.format(hline=hline)
  ]
  for i, model_name in enumerate(results):
    acc_train, acc_test = results[model_name]['accuracy_train'], results[model_name]['accuracy_test']

    acc_train_str = '%.3lf' % (acc_train, )
    acc_test_str = ('\\textbf{%.3lf}' if i == best_indx else '%.3lf') % (acc_test, )

    rows.append(
      '  {model} & {acc_train} & {acc_test} \\\\\n  {hline}'.format(
        model=model_name, acc_train=acc_test_str, acc_test=acc_train_str, hline=hline
      )
    )

  table = '\n'.join(rows)
  return '\\begin{{tabular}}{{|l | c | c|}}\n{table}\n\\end{{tabular}}'.format(table=table)

def as_table(results):
  rows = [
    ['model', 'accuracy test', 'accuracy train']
  ]
  for model in results:
    rows.append([
      model,
      '%.4lf' % (results[model]['accuracy_test'], ),
      '%.4lf' % (results[model]['accuracy_train'], )
    ])

  width = [
    max([len(row[i]) for row in rows]) + 2
    for i in range(3)
  ]

  return '\n'.join([
    ''.join([item.center(w) for item, w in zip(row, width)])
    for row in rows
  ])