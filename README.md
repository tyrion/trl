# trl

## Requirements
* Python3
* [ifqi](https://github.com/teopir/ifqi/)

## Running an Experiment

You can use the `run.py` script to run an experiment. It supports the FQI and
PBO (NES) algorithms..
```
$ python3 run.py --help
usage: run.py [-q {nn,nn2,curve_fit}] [-n N] [-t N] [-e N] [-h N] [-b N] [-r]
              [--timeit N] [-s SEED SEED] [--help]
              env {fqi,pbo,ifqi_fqi,ifqi_pbo}

positional arguments:
  env                   The environment to use. Either from ifqi or gym.
  {fqi,pbo,ifqi_fqi,ifqi_pbo}
                        The algorithm to run

optional arguments:
  -q {nn,nn2,curve_fit}
                        Q regressor to use
  -n N, --training-iterations N
                        number of training iterations. default is 50.
  -t N, --training-episodes N
                        Number of training episodes to collect.
  -e N, --evaluation-episodes N
                        Number of episodes to use for evaluation.
  -h N, --horizon N     Max number of steps per episode.
  -b N, --budget N      budget
  -r, --render          Render the environment during evaluation.
  --timeit N            Benchmark algorithm, using N repetitions
  -s SEED SEED, --seeds SEED SEED
                        specify the random seeds to be used (gym.env,
                        np.random)
  --help                show this help message and exit
```

Simple run:
```
KERAS_BACKEND=theano python3 run.py LQG1D-v0 fqi -q curve_fit
```

Perform a simple benchmark:
```
$ python3 run.py LQG1D-v0 ifqi_fqi -n 20 -q curve_fit --timeit 5
```
