import logging
import logging.config
import os
import sys

os.environ.setdefault('KERAS_BACKEND', 'theano')

import pytest
import theano

floatX = theano.config.floatX
import keras  # this will override theano.config.floatX

# respect theano settings.
keras.backend.set_floatx(floatX)
theano.config.floatX = floatX

import numpy as np
import numdifftools as nd
from theano import tensor as T

from ifqi import envs

from trl import algorithms, regressor, utils
from trl.algorithms import ifqi
from trl.experiment import Experiment

LOGGING = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)5s:%(name)s: %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
        },
    },
    'loggers': {
        'trl': {
            'level': 'DEBUG',
        },
    },
    'root': {
        'level': 'DEBUG',  # debug
        'handlers': ['console'],
    },
}

logging.config.dictConfig(LOGGING)


class FakeRequest():
    param = None


class CurveFitQRegressor(regressor.TheanoRegressor):
    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params - 0.0001)

    def Q(self, sa, b, k):
        return self.model(sa, [b, k])

    def _model(self, sa, theta):
        s = sa[:, 0]
        a = sa[:, 1]
        b = theta[0]
        k = theta[1]
        return - b * b * s * a - 0.5 * k * a * a - 0.4 * k * s * s


def build_nn(input_dim=2, output_dim=2, activation='sigmoid'):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import callbacks

    model = Sequential()
    model.add(Dense(2, input_dim=input_dim, init='uniform',
                    activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return regressor.KerasRegressor(model, input_dim)


class BaseExperiment(Experiment):
    env_name = 'LQG1D-v0'
    training_episodes = 5
    training_iterations = 2
    evaluation_episodes = 2
    budget = 1

    def get_q(self):
        return CurveFitQRegressor(np.array([0, 0], dtype=floatX))

    def get_algorithm_config(self):
        # bo needs to be created here due to seed settings.
        ndim = len(self.q.params)
        self.algorithm_config['bo'] = build_nn(ndim, ndim)
        return self.algorithm_config


@pytest.fixture
def experiment(request):
    opts, algo_c, summary = request.param
    print(algo_c, file=sys.stderr)
    Ex = type('Ex', (BaseExperiment,), opts)

    def get_experiment(algo):
        e = Ex(algorithm_class=algo, algorithm_config=algo_c.copy(),
               initial_states=None, horizon=50)
        return e, summary

    return get_experiment


def run(experiment, algorithm):
    e, summary = experiment(algorithm)
    r = e.run()
    assert np.allclose(summary, r[1]), "{} {}".format(summary, r[1])


nes_params = (
    [
        {'np_seed': 7094654038104888253, 'env_seed': 3729446728225797397},
        {'incremental': True, 'batch_size': 10, 'learning_rate': 0.05},
        [50., -1.19969833, -57.35381317]],
    [
        {'np_seed': 13594933323247414643, 'env_seed': 12280054697174909087},
        {'incremental': True, 'batch_size': 12, 'learning_rate': 0.2},
        [50., -1.66234112, -80.40379333]],
    [
        {'np_seed': 7094654038104888253, 'env_seed': 3729446728225797397},
        {'incremental': False, 'batch_size': 10, 'learning_rate': 0.1},
        [50., -1.19048798, -57.26132202]],
    [
        {'np_seed': 13594933323247414643, 'env_seed': 12280054697174909087},
        {'incremental': False, 'batch_size': 12, 'learning_rate': 2},
        [50., -1.55656433, -75.25311279]],
)


@pytest.mark.parametrize('experiment', nes_params, True)
def test_nespbo(experiment):
    run(experiment, algorithms.NESPBO)


@pytest.mark.parametrize('experiment', nes_params, True)
def test_nespbo_ifqi(experiment):
    run(experiment, ifqi.PBO)


grad_params = (
    [
        {'np_seed': 7094654038104888253, 'env_seed': 3729446728225797397},
        {'incremental': True, 'K': 1, 'update_index': 1, 'update_steps': 1,
         'batch_size': 10},
        [50., -1.08974481, -52.04819489]],
    [
        {'np_seed': 13594933323247414643, 'env_seed': 12280054697174909087},
        {'incremental': True, 'K': 2, 'update_index': 2, 'update_steps': 1,
         'batch_size': 5},
        [50., -1.66234112, -80.40379333]],
    [
        {'np_seed': 7094654038104888253, 'env_seed': 3729446728225797397},
        {'incremental': False, 'K': 3, 'update_index': 2, 'update_steps': 1,
         'batch_size': 7},
        [50., -1.68249202, -80.60005951]],
    [
        {'np_seed': 13594933323247414643, 'env_seed': 12280054697174909087},
        {'incremental': False, 'K': 1, 'update_index': 1, 'update_steps': 2,
         'batch_size': 1},
        [50., -5.69852829, -270.3427124]],
)


@pytest.mark.parametrize('experiment', grad_params, True)
def test_gradpbo(experiment):
    run(experiment, algorithms.GradPBO)


@pytest.mark.parametrize('experiment', grad_params, True)
def test_gradpbo_ifqi(experiment):
    run(experiment, ifqi.GradPBO)


@pytest.mark.parametrize("opts, algo_c, summary", grad_params)
def test_gradpbo_history_comparison(opts, algo_c, summary):
    request = FakeRequest()
    request.param = (opts, algo_c, summary)

    cexp = experiment(request)
    e, summary = cexp(ifqi.GradPBO)
    r1 = e.run()
    ifqig = e.algorithm

    cexp = experiment(request)
    e, summary = cexp(algorithms.GradPBO)
    r2 = e.run()
    trlg = e.algorithm

    assert np.allclose(r1[1], r2[1]), "{}, {}".format(r1[1], r2[1])

    t1 = np.array(ifqig.history['theta']).squeeze()
    t2 = np.array(trlg.history['theta']).squeeze()
    assert np.allclose(t1, t2), "{}, {}".format(t1, t2)

    for v1, v2 in zip(ifqig.history['rho'], trlg.history['rho']):
        for sv1, sv2 in zip(v1, v2):
            assert np.allclose(sv1, sv2), "{}, {}".format(sv1, sv2)


if __name__ == '__main__':
    st = FakeRequest()
    st.param = grad_params[3]
    test_gradpbo_history_comparison(*st.param)
    # cexp = experiment(st)
    # test_gradpbo(cexp)
    # cexp = experiment(st)
    # test_gradpbo_ifqi(cexp)
