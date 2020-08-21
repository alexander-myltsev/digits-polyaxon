# Digits

```info
author: M Borisyak, V Belavin, D Maevsky, N Kazeev, D Derkach, A Ustyuzhanin on behalf of LAMBDA
```

> Because there is not enough such tutorials.

## What is this?

This is a toy ML study (supposedly) made according to the laboratory guidelines. The purpose of this project is to demonstrate *one* of the good practices for organizing
ML-related studies.

Here, we train simple classifiers on:
- Google's Street View House Numbers;
- MNIST.

## Prerequisites

For this tutorial (and for all projects, in general) we highly recommend
setting up a new virtual environment with python 3.5+ and pip.

If you are unfamiliar with Python virtual environments, here are a few options:
- standard [venv](https://docs.python.org/3/tutorial/venv.html) and
  [virtualenvwrapper](https://pypi.org/project/virtualenvwrapper/);
- popular [conda](https://docs.conda.io/en/latest/);
- dead simple, yet effective [pyenv](https://github.com/pyenv/pyenv).

## Installation and usage

This repository is set up as a proper python package, thus it can be installed by:
```sh
git clone git@gitlab.com:lambda-hse/digits.git
cd digits

# installs requirements from the file
pip install -r requirements.txt

# installs the package
pip install -e .
```

First, if you make a fork, please, substitute `git@gitlab.com:lambda-hse/digits.git` with URL of your fork.

Second, notice absence of the dot (`.`) in the third command and its presence in the last one --- the third command installs packages from `requirements.txt`,
the fourth install the package from `.` directory (current directory).

Third, take into account `-e` flag in the last command --- it tells pip to install the package in "editable" mode:
all changes to the project files immediately take effect without need to reinstall the package (this, however, does not affect already imported code in a running interpreter).
Nevertheless, adding/removing top-level modules (ones directly under `src/`) still requires reinstallation.


## Run the code

It is good moment to [read a bit](https://gitlab.com/lambda-hse/digits/-/wikis/home) to get familiar with project structure, working with the code, running it and examining results.

## Ideas for improvement

Here are a few suggestions:
- launch an experiment, make report;
- implement your own architecture:
  - the following suggestions might be a good start:
    - introducing data augmentation;
    - adding dropout, batch norm;
    - implementing ResNet;
    - pretraining with the `extra` dataset (see, [SVHN home page](http://ufldl.stanford.edu/housenumbers/));
- introduce a new dataset (e.g.,
  [fasion MNIST](https://github.com/zalandoresearch/fashion-mnist),
  [not MNIST](http://yaroslavvb.com/upload/notMNIST/));
    - train all models on the new dataset and update the report;


