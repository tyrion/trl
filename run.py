#!/usr/bin/env python3
import argparse
import logging
import logging.config
import signal
import warnings

import numpy as np


from scipy.optimize import curve_fit

from trl import algorithms, ifqi, regressor
from trl.experiment import Experiment


class CurveFitQRegressor(regressor.Regressor):

    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params-0.0001)

    def Q(self, sa, b, k):
        s, a = sa[:, 0], sa[:, 1]
        return - b * b * s * a - 0.5 * k * a * a - 0.4* k * s * s

    def predict(self, x):
        return self.Q(x, *self.params)


def build_nn(activation='sigmoid', input_dim=2, output_dim=2):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import callbacks

    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, init='uniform',
                    activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return regressor.KerasRegressor(model, input_dim)


def build_nn2(activation='sigmoid', input_dim=2, output_dim=2):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import callbacks

    cb = callbacks.EarlyStopping(monitor='loss', min_delta=6e-1,
                                 patience=5, mode='auto')

    model = Sequential()
    model.add(Dense(4, input_dim=input_dim, init='uniform', activation=activation))
    model.add(Dense(4, init='uniform', activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return regressor.KerasRegressor(model, input_dim, callbacks=[cb],
                                    nb_epoch=50, batch_size=100)

def build_curve_fit(input_dim=2, output_dim=1):
    return CurveFitQRegressor(np.array([0,0]))


def handler(signum, frame):
    logger.critical('Received Interrupt. Terminating')
    raise SystemExit


class CLIExperiment(Experiment):

    def get_q(self, q):
        q = globals()['build_{}'.format(q)]
        return q(input_dim=self.input_dim, output_dim=1)

    def get_algorithm(self, algorithm, **kwargs):
        algo = {
            'fqi': algorithms.FQI,
            'pbo': algorithms.NESPBO,
            'ifqi_fqi': ifqi.FQI,
            'ifqi_pbo': ifqi.PBO,
        }[algorithm]

        if 'pbo' in algorithm:
            dim = len(self.q.params)
            kwargs['bo'] = build_nn2(input_dim=dim, output_dim=dim)

        return algo(self, **kwargs)



if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('env',
        help='The environment to use. Either from ifqi or gym.')
    parser.add_argument('algorithm',
        choices=['fqi', 'pbo', 'ifqi_fqi', 'ifqi_pbo'],
        help='The algorithm to run')
    parser.add_argument('-q', help='Q regressor to use',
        choices=['nn', 'nn2', 'curve_fit'], default='nn2')
    parser.add_argument('-n', '--training-iterations',
        metavar='N', type=int, default=50,
        help='number of training iterations. default is 50.')
    parser.add_argument('-t', '--training-episodes',
        metavar='N', type=int, default=100,
        help='Number of training episodes to collect.')
    parser.add_argument('-e', '--evaluation-episodes',
        metavar='N', type=int,
        help='Number of episodes to use for evaluation.')
    parser.add_argument('-h', '--horizon', type=int, metavar='N',
        help='Max number of steps per episode.')
    parser.add_argument('-b', '--budget', type=int, help='budget', metavar='N')
    parser.add_argument('-r', '--render', action='store_true',
        help='Render the environment during evaluation.')
    parser.add_argument('--timeit', type=int, default=0, metavar='N',
        help='Benchmark algorithm, using N repetitions')
    parser.add_argument('-s', '--seeds', type=int,
        nargs=2, default=[None, None], metavar='SEED',
        help='specify the random seeds to be used (gym.env, np.random)')
    parser.add_argument('--help', action='help',
        help='show this help message and exit')
    args = parser.parse_args()

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'format': '%(levelname)5s:%(name)s: %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': logging.INFO,
                'formatter': 'default',
            },
        },
        'loggers': {
            'trl': {
                'level': logging.DEBUG,
            },
        },
        'root': {
            'level': logging.INFO,
            'handlers': ['console'],
        },

    })
    logger = logging.getLogger('')

    # pybrain is giving a lot of deprecation warnings
    warnings.filterwarnings('ignore', module='pybrain')

    args = dict(vars(args))
    args['np_seed'], args['env_seed'] = args.pop('seeds')
    timeit = args.pop('timeit')

    experiment = CLIExperiment(**args)
    if timeit:
        experiment.benchmark(timeit)
    else:
        experiment.train()
    experiment.evaluate()
