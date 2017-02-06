# trl

[![Build Status](https://travis-ci.org/tyrion/trl.svg?branch=master)](https://travis-ci.org/tyrion/trl)

## Prerequisites

This project requires Python 3 to run.

## Install

Clone the repository:
```bash
$ git clone https://github.com/tyrion/trl.git
$ cd trl
```

Make a virtualenv to install dependencies (Optional):
```bash
$ mkvirtualenv -p /usr/bin/python3 -a `pwd` trl
```

Install the requirements:
```bash
$ pip install -r requirements.txt
```
Install the project:
```bash
$ pip install -e .
```

## Run the tests

Install the test dependencies:
```bash
$ pip install -e .[test]
```

Run the tests:
```bash
$ pytest
```

## Run an Experiment

You can use the `run.py` script to run an experiment. It supports the FQI and PBO algorithms.
```bash
$ experiments/run.py --help
usage: run.py [-n N] [-t N] [-e N] [-h N] [-b N] [-l FILEPATH]
              [--load-dataset FILEPATH] [-q FILEPATH] [-o FILEPATH]
              [--save-dataset FILEPATH] [--save-regressor FILEPATH]
              [--save-trace FILEPATH] [-r] [--timeit N] [-s SEED [SEED ...]]
              [--log-level LEVEL | -v | --quiet] [--help]
              env_name
              {fqi,pbo,ifqi_fqi,ifqi_pbo,gradfqi,gradpbo,ifqi_gradpbo}

positional arguments:
  env_name              The environment to use. Either from ifqi or gym.
  {fqi,pbo,ifqi_fqi,ifqi_pbo,gradfqi,gradpbo,ifqi_gradpbo}
                        The algorithm to run

optional arguments:
  -n N, --training-iterations N
                        number of training iterations. default is 50.
  -t N, --training-episodes N
                        Number of training episodes to collect.
  -e N, --evaluation-episodes N
                        Number of episodes to use for evaluation.
  -h N, --horizon N     Max number of steps per episode.
  -b N, --budget N      budget

io:
  Load/Save

  -l FILEPATH, --load FILEPATH
                        Load both the dataset and the Q regressor from
                        FILEPATH
  --load-dataset FILEPATH
                        Load the dataset from FILEPATH
  -q FILEPATH, --load-regressor FILEPATH
                        Load the trained Q regressor from FILEPATH. You can
                        also specifyone of {nn,nn2,curve_fit} instead of
                        FILEPATH.
  -o FILEPATH, --save FILEPATH
                        Save both the dataset and the Q regressor to FILEPATH
  --save-dataset FILEPATH
                        Save the dataset to FILEPATH
  --save-regressor FILEPATH
                        Save the trained Q regressor to FILEPATH
  --save-trace FILEPATH
                        Save the evaluation trace to FILEPATH

others:
  -r, --render          Render the environment during evaluation.
  --timeit N            Benchmark algorithm, using N repetitions
  -s SEED [SEED ...], --seeds SEED [SEED ...]
                        specify the random seeds to be used (gym.env,
                        np.random)
  --log-level LEVEL     Set loglevel to LEVEL
  -v, --verbose         Show more output. Same as --log-level DEBUG
  --quiet               Show less output. Same as --log-level ERROR
  --help                show this help message and exit

```

Simple run:
```
$ experiments/run.py LQG1D-v0 fqi -q curve_fit
```

Perform a simple benchmark:
```
$ experiments/run.py LQG1D-v0 ifqi_fqi -n 20 -q curve_fit --timeit 5
```
