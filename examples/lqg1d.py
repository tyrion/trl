
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


from ifqi import envs
from ifqi.evaluation import evaluation

from trl.evaluation import collect_episodes, QPolicy
from trl import algorithms, regressor, utils


states = np.linspace(-10, 10, 20)
actions = np.linspace(-8, 8, 20)
initial_states = np.array([[1, 2, 5, 7, 10]]).T


class CurveFitQRegressor(regressor.Regressor):

    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params-0.0001)

    def Q(self, sa, b, k):
        s, a = sa[:, 0], sa[:, 1]
        return - b * b * s * a - 0.5 * k * a * a - 0.4* k * s * s

    def predict(self, x):
        return self.Q(x, *self.params)

    def evaluate(self, X, Q_fun):
        Q_hat = self.predict(X)
        Q_opt = Q_fun(X)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], Q_opt)
        ax.scatter(X[:, 0], X[:, 1], Q_hat, c='r', marker='^')

        mae = mean_absolute_error(Q_opt, Q_hat)
        mse = mean_squared_error(Q_opt, Q_hat)

        print("       ( mse {:8.2f}, mae {:8.2f})".format(mse, mae))
        print(" theta (   b {:8.2f},   k {:8.2f})".format(*self.params))

        evaluateQ(self.predict)
        return Q_hat


def init_env(seed):
    env = envs.LQG1D()
    seed = env.seed(seed)
    print('Random seed1: {}'.format(seed))
    env.reset()
    dataset = collect_episodes(env, n=100)

    return env, dataset


def compute_optimal(env):
    K, cov = env.computeOptimalK(), 0.001
    print('Optimal K: {} Covariance S: {}'.format(K[0][0], cov))

    Q_fun_ = np.vectorize(lambda s, a: env.computeQFunction(s, a, K, cov, 1))
    Q_fun = lambda X: Q_fun_(X[:, 0], X[:, 1])
    optimalP = QPolicy(Q_fun, actions)
    return K, optimalP


def evaluateP(policy, i=initial_states):
    values = evaluation.evaluate_policy(env, policy, initial_states=i)
    print("values (mean {:8.2f},  se {:8.2f})\n steps (mean {:8.2f},  se {:8.2f})".format(*values))
    return values


def evaluateQ(Q, i=initial_states):
    return evaluateP(QPolicy(Q, actions), i)


def build_nn(activation='sigmoid', input_dim=2):
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, init='uniform',
                    activation=activation))
    model.add(Dense(2, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return regressor.KerasRegressor(model, input_dim)


def setup_pbo():
    q = CurveFitQRegressor(np.array([0.0, 0.0]))
    bo = build_nn()
    return algorithms.NESPBO(dataset, actions, q, bo, env.gamma, K=1,
                            batchSize=10, learningRate=0.01)


def setup_fqi():
    q = CurveFitQRegressor(np.array([0.0, 0.0]))
    return algorithms.FQI(dataset, actions, q, env.gamma)


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

    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', help='The algorithm to run',
                        choices=['fqi', 'pbo'])
    parser.add_argument('-n', type=int, default=50,
        help='number of iterations. default is 50.')
    parser.add_argument('-b', '--budget', type=int, help='budget')
    parser.add_argument('-p', '--plot', help='plot results', action='store_true')
    parser.add_argument('-t', '--timeit', type=int, default=0)
    parser.add_argument('--seed1', type=int, help='specify a random seed.')
    parser.add_argument('--seed2', type=int, help='seed numpy')
    args = parser.parse_args()

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'format': '%(levelname)s:%(name)s: %(message)s',
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

    seed = seeding._seed(args.seed2)
    np.random.seed(seeding._int_list_from_bigint(seeding.hash_seed(seed)))

    env, dataset = init_env(args.seed1)
    print('Random seed2: {}'.format(seed))

    setup = locals()['setup_{}'.format(args.algorithm)]
    algorithm = setup()
    if args.timeit:
        t = timeit.repeat('algorithm.run(args.n, args.budget)',
                          number=1, repeat=args.timeit, globals=globals())
        print('{} iterations, best of {}: {}s\n'.format(
                args.n, args.timeit, min(t)))
    else:
        algorithm.run(args.n, args.budget)

    print('algorithm finished.')
    SA = utils.make_grid(states, actions)

    K, optimalP = compute_optimal(env)
    print('\noptimal:')
    evaluateP(optimalP)

    print('\nlearned:')
    algorithm.q.evaluate(SA, optimalP.q)

    if args.plot:
        plt.show()


