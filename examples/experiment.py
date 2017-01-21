#!/usr/bin/env python3

import gym
import numpy as np

from gym import spaces
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


from ifqi import envs
from ifqi.evaluation import evaluation

from trl import algorithms, evaluation, regressor, utils




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

    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, init='uniform',
                    activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return regressor.KerasRegressor(model, input_dim)


def build_nn2(activation='sigmoid', input_dim=2, output_dim=2):
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(5, input_dim=input_dim, init='uniform', activation=activation))
    model.add(Dense(5, init='uniform', activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return regressor.KerasRegressor(model, input_dim)



def setup_pbo(env, dataset, actions, q, args):
    bo = build_nn()
    return algorithms.NESPBO(dataset, actions, q, bo, gamma=0.99, K=1,
                            batchSize=10, learningRate=0.1).run


def setup_fqi(env, dataset, actions, q, args):
    return algorithms.FQI(dataset, actions, q, gamma=0.99).run



def setup_ifqi_pbo(env, dataset, actions, q, args):
    from ifqi.algorithms.pbo.pbo import PBO
    from ifqi.models.regressor import Regressor

    r = Regressor(object)
    r._regressor = q

    bo = build_nn()
    state_dim, action_dim, reward_dim = envs.get_space_info(env)

    pbo = PBO(estimator=r,
          estimator_rho=bo.model,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=actions,
          gamma=env.gamma,
          learning_steps=args.n,
          batch_size=10,
          learning_rate=0.1,
          incremental=False,
          verbose=False)

    sast = utils.rec_to_array(dataset[
        ['state', 'action', 'next_state', 'absorbing']])
    r = dataset.reward

    return lambda *args: pbo.fit(sast, r)


def get_env(name):
    try:
        return getattr(envs, name)()
    except AttributeError:
        return gym.make(name)


def discretize_space(space: gym.Space, max=20):
    if isinstance(space, spaces.Discrete):
        return np.arange(space.n)

    if isinstance(space, spaces.Box):
        # only 1D Box supported
        return np.linspace(space.low, space.high, max)


def get_space_dim(space: gym.Space):
    return np.prod(getattr(space, 'shape', 1))



if __name__ == '__main__':
    import argparse
    import logging
    import logging.config
    import random
    import signal
    import time
    import timeit
    import warnings

    from gym.utils import seeding

    def handler(signum, frame):
        print('Received Interrupt. Terminating')
        raise SystemExit

    signal.signal(signal.SIGINT, handler)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('env',
        help='The environment to use. Either from ifqi or gym.')
    parser.add_argument('algorithm', choices=['fqi', 'pbo', 'ifqi_pbo'],
        help='The algorithm to run')
    parser.add_argument('-n', type=int, default=50,
        help='number of learning iterations. default is 50.')
    parser.add_argument('-e', '--episodes', type=int, default=100,
        help='Number of training episodes to collect.')
    parser.add_argument('-h', '--horizon', type=int,
        help='Max number of steps per training episode.')
    parser.add_argument('-b', '--budget', type=int, help='budget')
    parser.add_argument('-p', '--plot', help='plot results', action='store_true')
    parser.add_argument('-t', '--timeit', type=int, default=0)
    parser.add_argument('-s', '--seeds', type=int, nargs=2, default=[None, None],
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

    # pybrain is giving a lot of deprecation warnings
    warnings.filterwarnings('ignore', module='pybrain')

    # seeding np.random (for pybrain)
    seed = seeding._seed(args.seeds[1])
    np.random.seed(seeding._int_list_from_bigint(seeding.hash_seed(seed)))

    env = get_env(args.env)
    seeds = env.seed(args.seeds[0])
    seeds.append(seed)
    env.reset()
    print('Random seeds:', ' '.join(str(x) for x in seeds))
    print('Initialized env', args.env)
    print('observation space: ', env.observation_space)
    print('action space: ', env.action_space)

    actions = discretize_space(env.action_space)
    print('Discretized actions ({}): {}'.format(len(actions), actions))

    # collect training data
    horizon = args.horizon or getattr(env, 'horizon', 100)
    print('Collecting training data (episodes: {}, horizon: {})'.format(
        args.episodes, horizon), end='')
    dataset, _ = evaluation.interact(env, args.episodes, horizon, collect=True)
    env.reset()
    print(' done')



    print('Training algorithm')
    # XXX could try to start from 1,0
    #q = CurveFitQRegressor(np.array([1.0, 0.0]))
    input_dim = get_space_dim(env.observation_space) + \
                get_space_dim(env.action_space)
    q = build_nn2(input_dim=input_dim, output_dim=1)

    setup = locals()['setup_{}'.format(args.algorithm)]
    algorithm = setup(env, dataset, actions, q, args)
    if args.timeit:
        t = timeit.repeat('algorithm(args.n, args.budget)',
                          number=1, repeat=args.timeit, globals=globals())
        print('{} iterations, best of {}: {}s\n'.format(
                args.n, args.timeit, min(t)))
    else:
        algorithm(args.n, args.budget)

    print('Training finished.')

    policy = evaluation.QPolicy(q, actions)
    evaluation.interact(env, 10, 500, policy, render=True,
                        metrics=[evaluation.discounted(0.9)])
