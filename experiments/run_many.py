#!/usr/bin/env python3
import argparse
import itertools
import logging
import logging.config
import os
import signal

import base


import numpy as np
# register ifqi's envs
from ifqi import envs

from trl import algorithms, experiment, regressor, utils


signal.signal(signal.SIGINT, base.handler)
logging.config.dictConfig(base.LOGGING)

import keras


class Experiment(base.CLIExperiment):
    budget = 1
    timeit = 1

    @classmethod
    def run_ith(cls, i, **kwargs):
        kwargs['np_seed'], kwargs['env_seed'] = seeds[i]
        return super().run_ith(i, **kwargs)


def make_dataset(i, **config):
    logging.disable(logging.INFO)
    e = Experiment(training_iterations=0, evaluation_episodes=0,
                   training_episodes=100,
                   dataset_save_path='dataset-{}.h5'.format(i), **config)
    e.run()
    logging.disable(logging.NOTSET)
    return (e.np_seed, e.env_seed)


def pbo_params():
    experiments = list(itertools.product([1,5,10,20], [1,2,5,10], [1,2,5]))
    experiments_ = list(itertools.product(
        [1,5,10,20], # iterations
        [1,2,5,10],  # K
        [0, 1],      # incremental
        [1, 10, 100],# batch_size
        [2, np.inf], # norm_value
    ))
    name_fmt = 'n{:02}K{:02}B{:02}'

    for (n, K, B) in experiments:
        name = name_fmt.format(n, K, B)
        config = {'training_iterations': n, 'budget': B}
        algo_config = {'K': K}
        yield name, config, algo_config


def gradpbo_params():
    experiments = list(itertools.product(
        [1, 2, 3, 4],  # iterations
        [1, 2, 5, 10], # K
        [0, 1],        # incremental
        [0],           # independent
        [1, 5, 10],    # update_index
        [1, 10, 100],  # batch_size
        [2, np.inf],   # norm_value
    ))
    name_fmt = 'n{:02}K{:02}inc{}ind{}ui{:02}us{:02}b{:02}no{:03}'

    for (n, K, inc, ind, u_i, batch, norm) in experiments:
        u_s = K
        name = name_fmt.format(n, K, inc, ind, u_i, u_s, batch, norm)
        config = {'training_iterations': n}

        algo_config = {
            'K': K,
            'incremental': inc,
            'independent': ind,
            'update_index': u_i,
            'update_steps': u_s,
            'batch_size': batch,
            'norm_value': norm,
        }
        yield name, config, algo_config


def fqi_params():
    experiments = list(range(1, 21))
    for n in experiments:
        yield 'n{:02}'.format(n), {'training_iterations': n}, {}

ifqi_fqi_params = fqi_params


parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('env_name',
    help='The environment to use. Either from ifqi or gym.')
parser.add_argument('algorithm',
    choices=['fqi', 'pbo', 'ifqi_fqi', 'ifqi_pbo', 'gradfqi', 'gradpbo',
             'ifqi_gradpbo'],
    help='The algorithm to run')
parser.add_argument('--repeat', type=int, default=10,
    help='Number of experiments to run.')
parser.add_argument('--create-dataset', action='store_true')
parser.add_argument('--no-run', action='store_true')
parser.add_argument('--workers', metavar='N', type=int,
    help='Number of parallel workers to execute')
parser.add_argument('--resume', action='store_true',
    help='Resume where you left off')
args = parser.parse_args()

N = args.repeat

if args.create_dataset:
    logging.info('Generating datasets')
    seeds = Experiment.run_many(N, make_dataset, env_name=args.env_name,
                                workers=args.workers)

    ordered_seeds = np.zeros((N, 2))
    for i in range(N):
        ordered_seeds[i] = seeds[i]
    utils.save_dataset(ordered_seeds, 'seeds.h5')
else:
    seeds = utils.load_dataset('seeds.h5')

if args.no_run:
    raise SystemExit


algo = args.algorithm
cls = base.ALGORITHMS[algo]

os.makedirs(algo, exist_ok=True)

get_params = globals()['{}_params'.format(algo)]
for (name, config, algo_config) in get_params():
    path = os.path.join(algo, name)

    if args.resume and os.path.exists(path):
        logging.info('=== Skipping %s', name)
        continue

    os.makedirs(path, exist_ok=True)
    logging.info('=== Executing %s', name)
    r = Experiment.run_many(N,
            env_name=args.env_name,
            algorithm_class=cls,
            save_path=os.path.join(path, 'experiment-{i}.h5'),
            dataset_load_path='dataset-{i}.h5',
            dataset_save_path=None,
            logfile=os.path.join(path, 'experiment-{i}.log'),
            algorithm_config=algo_config,
            workers=args.workers,
            **config)

    time, summary = list(zip(*r.values()))
    logging.info("=== Batch Summary: %f %s", np.mean(time), np.mean(summary, 0))

