#!/usr/bin/env python3
import argparse
import copy
import itertools
import logging
import logging.config
import os
import shutil

import base
import postprocess


import numpy as np



# register ifqi's envs
from ifqi import envs

from trl import algorithms, experiment, regressor, utils



import keras


class Experiment(base.CLIExperiment):
    budget = 1
    timeit = 1

    @classmethod
    def run_ith(cls, i, seeds, **kwargs):
        kwargs['np_seed'], kwargs['env_seed'] = seeds[i]
        return super().run_ith(i, **kwargs)


def make_dataset(i, **config):
    logging.disable(logging.INFO)
    e = Experiment(training_iterations=0, evaluation_episodes=0,
                   training_episodes=1000,
                   dataset_save_path='dataset-{}.h5'.format(i), **config)
    e.run()
    logging.disable(logging.NOTSET)
    return (e.np_seed, e.env_seed)


def pbo_params():
    experiments = list(itertools.product([1,5,10,20], [1,2,5,10], [1,2,5]))
    experiments_ = list(itertools.product(
        [1,5,10,20], # iterations
        [1,2,5,10],  # K

    ))
    exp = [
    #   n   K   B
        (50, 1, 1),
        (25, 2, 1),
        (25, 1, 2),
        (25, 2, 2),
        (10, 1, 5),
        (10, 5, 1),
        (10, 2, 2),
    ]

    config = list(itertools.product(
        [0, 1],      # incremental
        [1, 10, 100],# batch_size
        [2, np.inf], # norm_value
    ))
    name_fmt = 'n{:02}K{:02}B{:02}'

    for (inc, batch, norm) in config:
        pass


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
    experiments = [
        (50 , 1, 0, 0, 1, 100, np.inf),
        (50 , 1, 0, 0, 1, 100, 2),
        (50 , 1, 1, 0, 1, 100, np.inf),
        (50 , 1, 1, 0, 1, 100, 2),

        (50 , 1, 0, 0, 1, 10, np.inf),
        (50 , 1, 0, 0, 1, 10, 2),
        (50 , 1, 1, 0, 1, 10, np.inf),
        (50 , 1, 1, 0, 1, 10, 2),

        (100 , 1, 0, 0, 1, 100, np.inf),
        (100 , 1, 0, 0, 1, 100, 2),
        (100 , 1, 1, 0, 1, 100, np.inf),
        (100 , 1, 1, 0, 1, 100, 2),

        (100 , 1, 0, 0, 1, 10, np.inf),
        (100 , 1, 0, 0, 1, 10, 2),
        (100 , 1, 1, 0, 1, 10, np.inf),
        (100 , 1, 1, 0, 1, 10, 2),
    ]
    experiments = list(itertools.product(
        [None, 'auto', 'be'], # update_loss
        [np.inf, 2],   # norm_value
        [50, 100, 150],  # iterations
        [1], # K
        [0, 1],        # incremental
        [0],           # independent
        [1, 5, 10],    # update_index
        [1, 5, 10],    # update_steps
        [100],  # batch_size
    )) +  list(itertools.product(
        ['auto', 'be'],
        [np.inf, 2],
        [50, 100, 150],
        [1],
        [0],
        [0],
        [1, 5, 10],
        [200],
        [100],
    ))
    experiments = [
        #(None, 2, 500, 1, 0, 0, 5, 5, 100)
        #(None, 2, 500, 1, 0, 0, 10, 20, 100)
        (None, 2, 500, 1, 0, 0, 1, 1, 32)
    ]
    name_fmt = 'n{:02}K{:02}inc{}ind{}ui{:02}us{:02}b{:02}no{:03}ul{}'

    for (u_l, norm, n, K, inc, ind, u_i, u_s, batch) in experiments:
        name = name_fmt.format(n, K, inc, ind, u_i, u_s, batch, norm, u_l)
        config = {
            'training_iterations': n,
            'evaluation_episodes': 289,
            'q_load_path': 'nfqi',
            'use_action_regressor': False,
        }

        algo_config = {
            'K': K,
            'incremental': inc,
            'independent': ind,
            'update_index': u_i,
            'update_steps': u_s,
            'batch_size': batch,
            'norm_value': norm,
            'update_loss': u_l,
        }
        yield name, config, algo_config



def fqi_params():
    experiments = list(range(1, 21))
    experiments = [50]
    for n in experiments:
        yield 'n{:02}'.format(n), {
            'training_iterations': n,
            'q_load_path': 'nn4',
            'evaluation_episodes': 289,
            'use_action_regressor': True,
        }, {}

ifqi_fqi_params = fqi_params

def wfqi_params():
    for steps in (0, 5, 20):
        yield 'n50ws%d' % steps, {
            'training_iterations': 50,
            'q_load_path': 'nfqi',
            'evaluation_episodes': 289,
            'use_action_regressor': False,
        }, {'weight_steps': 50}


def run_many(params, repeat=20, workers=None, resume=False, seeds=None,
             experiment_class=Experiment, base_path='experiment', **config):
    if seeds is None:
        seeds = [(None, None)] * repeat

    os.makedirs(base_path, exist_ok=True)

    for (name, cfg, algo_config) in params():
        path = os.path.join(base_path, name)

        if resume and os.path.exists(path):
            logging.info('=== Skipping %s', name)
            continue

        cfg.update(config)
        cfg.setdefault('save_path', os.path.join(path, 'experiment-{i}.h5'))
        cfg.setdefault('dataset_load_path', 'dataset-{i}.h5')
        cfg.setdefault('dataset_save_path', None)
        cfg.setdefault('logfile', os.path.join(path, 'experiment-{i}.log'))
        cfg.setdefault('workers', workers)
        cfg['seeds'] = seeds
        cfg['algorithm_config'] = algo_config

        os.makedirs(path, exist_ok=True)
        logging.info('=== Executing %s', name)
        r = experiment_class.run_many(repeat, **cfg)
        if r:
            time, summary = list(zip(*r.values()))
            logging.info("=== Batch Summary: %f %s", np.mean(time), np.mean(summary, 0))


def run_experiment(name, config, repeat=20, workers=None, resume=False,
                   seeds=None, experiment_class=Experiment, base_path='experiment'):

    path = os.path.join(base_path, name)
    if resume and os.path.exists(path):
        logging.info('=== Skipping %s', path)
        return postprocess.experiment_runs(path)

    cfg = copy.deepcopy(config)
    cfg.setdefault('save_path', os.path.join(path, 'experiment-{i}.h5'))
    cfg.setdefault('dataset_load_path', 'dataset-{i}.h5')
    cfg.setdefault('dataset_save_path', None)
    cfg.setdefault('logfile', os.path.join(path, 'experiment-{i}.log'))
    cfg.setdefault('workers', workers)
    cfg['seeds'] = seeds

    r = None
    try:
        os.makedirs(path, exist_ok=True)
        logging.info('=== Executing %s', name)
        r = experiment_class.run_many(repeat, **cfg)

        if r:
            time, summary = list(zip(*r.values()))
            logging.info("=== Batch Summary: %f %s", np.mean(time), np.mean(summary, 0))
            r = np.array([[r[k][1][-1], r[k][0]] for k in sorted(r)])
        return r
    finally:
        if r is None:
            logging.critical('deleting %s', path)
            shutil.rmtree(path)


def get_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('env_name',
        help='The environment to use. Either from ifqi or gym.')
    parser.add_argument('algorithm',
        choices=['fqi', 'pbo', 'wfqi', 'ifqi_fqi', 'ifqi_pbo', 'gradfqi', 'gradpbo',
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
    return parser


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGINT, base.handler)
    logging.config.dictConfig(base.LOGGING)

    parser = get_parser()
    args = parser.parse_args()

    N = args.repeat

    if args.create_dataset:
        logging.info('Generating datasets')
        seeds = Experiment.run_many(N, make_dataset, env_name=args.env_name,
                                    workers=args.workers)

        ordered_seeds = np.zeros((N, 2), 'uint64')
        for i in range(N):
            ordered_seeds[i] = seeds[i]
        utils.save_dataset(ordered_seeds, 'seeds.h5')
    else:
        seeds = utils.load_dataset('seeds.h5')

    if args.no_run:
        raise SystemExit

    args.algorithm_class = base.ALGORITHMS[args.algorithm]
    get_params = globals()['{}_params'.format(args.algorithm)]
    run_many(base_path=args.algorithm, params=get_params, seeds=seeds, **vars(args))
