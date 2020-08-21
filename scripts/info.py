from digits_utils import available

def info(target):
  print(' '.join(getattr(available, target)))

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
    prog='info',
    description='provides information on available experiments, datasets and models'
  )
  parser.add_argument('target', type=str, choices=available.__all__)
  arguments = parser.parse_args()

  info(arguments.target)
