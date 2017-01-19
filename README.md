# trl

## Requirements
* Python3
* [ifqi](https://github.com/teopir/ifqi/)

## Running an Example

The script supports running the FQI and PBO (NES) algorithms.
```
$ python3 examples/lqg1d.py --help
usage: lqg1d.py [-h] [-n N] [-b BUDGET] [-p] [-t TIMEIT] [-s SEEDS SEEDS]
                {fqi,pbo,ifqi_pbo}

positional arguments:
  {fqi,pbo,ifqi_pbo}    The algorithm to run

optional arguments:
  -h, --help            show this help message and exit
  -n N                  number of iterations. default is 50.
  -b BUDGET, --budget BUDGET
                        budget
  -p, --plot            plot results
  -t TIMEIT, --timeit TIMEIT
  -s SEEDS SEEDS, --seeds SEEDS SEEDS
                        specify the random seeds to be used (gym.env,
                        np.random)
```

Simple run:
```
$ python3 examples/lqg1d.py fqi
Random seeds: 9426525360237819434 12269849774062103771
algorithm finished.
Optimal K: -0.6152512456630115 Covariance S: 0.001

optimal:
values (mean   -66.20,  se    47.62)
 steps (mean   100.00,  se     0.00)

learned:
       ( mse 60839.05, mae   165.16)
 theta (   b    -2.83,   k    11.56)
values (mean   -67.06,  se    48.33)
 steps (mean   100.00,  se     0.00)
```

Perform a simple benchmark:
```
$ python3 examples/lqg1d.py ifqi_pbo -n 20 --budget 1 --timeit 5
Random seeds: 2280251555321456322 249365982371015718
Using TensorFlow backend.
20 iterations, best of 5: 6.972747469000751s

algorithm finished.
Optimal K: -0.6152512456630115 Covariance S: 0.001

optimal:
values (mean   -66.23,  se    47.70)
 steps (mean   100.00,  se     0.00)

learned:
       ( mse 18973.17, mae    88.68)
 theta (   b    -2.47,   k     2.20)
values (mean -4539.46,  se    49.18)
 steps (mean   100.00,  se     0.00)
```
