import logging
import logging.config
import os

os.environ.setdefault('KERAS_BACKEND', 'theano')

import pytest
import theano

floatX = theano.config.floatX
import keras # this will override theano.config.floatX

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
	'level': 'DEBUG', # debug
	'handlers': ['console'],
    },
}

logging.config.dictConfig(LOGGING)


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
    training_episodes = 10
    training_iterations = 2
    budget = 1

    def get_q(self):
        return CurveFitQRegressor(np.array([0, 0], dtype=floatX))

    def get_algorithm_config(self):
        # bo needs to be created here due to seed settings.
        ndim = len(self.q.params)
        self.algorithm_config['bo'] = build_nn(ndim, ndim)
        return self.algorithm_config



params = [
    (7094654038104888253, 3729446728225797397,
        np.array([100., -0.76798844, -68.44076508])),
]
import sys

@pytest.fixture
def experiment(request):
    opts, algo_c, summary = request.param
    sys.stderr.write(str(algo_c))
    sys.stderr.write('\n')
    Ex = type('Ex', (BaseExperiment,), opts)

    def get_experiment(algo):
        e = Ex(algorithm_class=algo, algorithm_config=algo_c.copy(), horizon=10)
        return e, summary

    return get_experiment


def run(experiment, algorithm):
    e, summary = experiment(algorithm)
    r = e.run()
    assert np.allclose(summary, r[1])

# def test_pbo(experiment):
#     e, summary = experiment(algorithms.NESPBO)
#     r = e.run()
#     assert np.allclose(summary, r[1])
#
#
# def test_ifqi_pbo(experiment):
#     e, summary = experiment(ifqi.PBO)
#     r = e.run()
#     assert np.allclose(summary, r[1])
#

def mk_params(keys, config_list):
    for config in config_list:
        summary = config.pop()
        params = [dict(zip(k, v)) for k, v in zip(keys, config)]
        params.append(summary)
        yield params


grad_params = list(mk_params([
    ('np_seed', 'env_seed'),
    ('incremental', 'K', 'update_index', 'update_steps', 'batch_size')
],[
#[ (7094654038104888253, 3729446728225797397), (True, 1, 1, 1, 10),
#  [ 100., -0.75184202, -67.01778412]],
[ (1312312, 2236864), (True, 1, 2, 1, 25),
  [10., -5.73367834, -57.0005722 ]],
[ (1312312, 2236864), (True, 1, 1, 1, 10),
  [10., -5.90091419, -58.89385986]],
[ (1312312, 2236864), (True, 3, 1, 1, 25), # lowering batch size it gets worse
  [10., -5.64493275, -56.3213501 ]],
]))


@pytest.mark.parametrize('experiment', grad_params, True)
def test_uegradpbo(experiment):
    run(experiment, algorithms.GradPBO)

@pytest.mark.parametrize('experiment', grad_params, True)
def test_ifqi_gradpbo(experiment):
    run(experiment, ifqi.GradPBO)
