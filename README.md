# trl

## Requirements
* Python3
* [ifqi](https://github.com/teopir/ifqi/)

## Running an Example

The script supports running the FQI and PBO (NES) algorithms.
```
usage: experiment.py [-n N] [-e EPISODES] [-h HORIZON] [-b BUDGET] [-r]
                     [-t TIMEIT] [-s SEEDS SEEDS] [--help]
                     env {fqi,pbo,ifqi_pbo}

positional arguments:
  env                   The environment to use. Either from ifqi or gym.
  {fqi,pbo,ifqi_pbo}    The algorithm to run

optional arguments:
  -n N                  number of learning iterations. default is 50.
  -e EPISODES, --episodes EPISODES
                        Number of training episodes to collect.
  -h HORIZON, --horizon HORIZON
                        Max number of steps per training episode.
  -b BUDGET, --budget BUDGET
                        budget
  -r, --render          Render the environment during evaluation.
  -t TIMEIT, --timeit TIMEIT
  -s SEEDS SEEDS, --seeds SEEDS SEEDS
                        specify the random seeds to be used (gym.env,
                        np.random)
  --help                show this help message and exit
```

Simple run:
```
KERAS_BACKEND=theano python3 examples/experiment.py LQG1D fqi -n 1 --render
```

Perform a simple benchmark:
```
$ python3 examples/experiment.py LQG1D ifqi_pbo -n 20 --budget 1 --timeit 5
```
