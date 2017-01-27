import logging
import logging.config
import os
import signal


import numpy as np
# register ifqi's envs
from ifqi import envs

from trl import algorithms, experiment, regressor

import base

signal.signal(signal.SIGINT, base.handler)
logging.config.dictConfig(base.LOGGING)

import keras



class Experiment(base.CLIExperiment):
    env_name = 'LQG1D-v0'
    algorithm_class = algorithms.NESPBO
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
                   dataset_save_path='dataset-{}.h5'.format(i))
    e.run()
    logging.disable(logging.NOTSET)
    return (e.np_seed, e.env_seed)

logging.info('Generating datasets')
seeds = Experiment.run_many(4, make_dataset)
#print(seeds)

experiments = [
#   n    K  B
#     (20, 1, 1),
#     (10, 1, 1),
#     (20, 1, 5),
#     (10, 2, 1),
#     (10, 2, 2),
    (2,  1, 1),
    (2,  5, 1),
    (2,  2, 5),
]

for (n, K, B) in experiments:
    logging.info('=== Executing with parameters n: %d, K: %d, B: %d', n, K, B)
    name = 'n{:02}K{:02}B{:02}'.format(n, K, B)
    os.makedirs(name, exist_ok=True)

    r = Experiment.run_many(4,
            training_iterations=n,
            save_path=os.path.join(name, 'experiment-{i}.h5'),
            dataset_load_path='dataset-{i}.h5',
            dataset_save_path=None,
            logfile=os.path.join(name, 'experiment-{i}.log'),
            budget=B,
            algorithm_config={'K': K})

    time, summary = list(zip(*r.values()))

    logging.info("=== Batch Summary: %f %s", np.mean(time), np.mean(summary, 0))
