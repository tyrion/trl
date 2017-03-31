import sys

import gym
import pytest
import numpy as np
import numdifftools as nd
import theano
from theano import tensor as T

from ifqi import envs

from trl import algorithms, regressor, utils
from trl.algorithms import ifqi
from trl.experiment import Experiment


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


def build_q(input_dim, output_dim=1):
    return CurveFitQRegressor(np.array([0, 0], dtype=theano.config.floatX))


def build_bo(q):
    ndim = len(q.params)
    return build_nn(ndim, ndim)


@pytest.fixture
def exp(request):
    seed, algo_c, summary = request.param
    return 1


def run(algo, seed, **config):
    config['bo'] = build_bo
    spec = gym.spec('LQG1D-v0')
    e = Experiment(spec, horizon=50, seed=seed)
    e.log_config()
    e.collect(episodes=5)
    e.train(q=build_q, iterations=2, stage=(3,1), algorithm_class=algo, algorithm_config=config)
    e.evaluate(policy=lambda e: e.policy, episodes=2)
    return e


nes_params = (
    [
        130873267332430886774782770234641364245,
        {'incremental': True, 'batch_size': 10, 'learning_rate': 0.05},
        [50., -1.19969833, -57.35381317]],
    [
        250782255713090746099536169063937622175,
        {'incremental': True, 'batch_size': 12, 'learning_rate': 0.2},
        [50., -1.66234112, -80.40379333]],
    [
        130873267332430886774782770234641364245,
        {'incremental': False, 'batch_size': 10, 'learning_rate': 0.1},
        [50., -1.19048798, -57.26132202]],
    [
        250782255713090746099536169063937622175,
        {'incremental': False, 'batch_size': 12, 'learning_rate': 2},
        [50., -1.55656433, -75.25311279]],
)


@pytest.mark.parametrize('seed,algo_c,summary', nes_params)
def test_nespbo(seed, algo_c, summary):
    e = run(algorithms.NESPBO, seed, budget=1, **algo_c)
    assert np.allclose(summary, e.summary)



@pytest.mark.parametrize('seed,algo_c,summary', nes_params)
def test_nespbo_ifqi(seed, algo_c, summary):
    e = run(ifqi.PBO, seed, **algo_c)
    assert np.allclose(summary, e.summary)


grad_params = (
    [
        130873267332430886774782770234641364245,
        {'incremental': True, 'K': 1, 'update_index': 1, 'update_steps': 1,
         'batch_size': 10},
        [50., -1.08974481, -52.04819489]],
    [
        250782255713090746099536169063937622175,
        {'incremental': True, 'K': 2, 'update_index': 2, 'update_steps': 1,
         'batch_size': 5},
        [50., -1.66234112, -80.40379333]],
    [
        130873267332430886774782770234641364245,
        {'incremental': False, 'K': 3, 'update_index': 2, 'update_steps': 1,
         'batch_size': 7},
        [50., -1.68249202, -80.60005951]],
    [
        109068179529540077892231880948389200726,
        {'incremental': False, 'K': 1, 'update_index': 1, 'update_steps': 2,
         'batch_size': 1},
        [50., -4.35781288, -206.3875885]],
)


@pytest.mark.parametrize('seed,algo_c,summary', grad_params)
def test_gradpbo(seed, algo_c, summary):
    e = run(algorithms.GradPBO, seed, **algo_c)
    assert np.allclose(summary, e.summary)


@pytest.mark.parametrize('seed,algo_c,summary', grad_params)
def test_gradpbo(seed, algo_c, summary):
    e = run(ifqi.GradPBO, seed, **algo_c)
    assert np.allclose(summary, e.summary)


@pytest.mark.parametrize('seed,algo_c,summary', grad_params)
def test_history(seed, algo_c, summary):
    e1 = run(ifqi.GradPBO, seed, **algo_c)
    e2 = run(algorithms.GradPBO, seed, **algo_c)
    ifqig = e1.algorithm
    trlg = e2.algorithm

    assert np.allclose(e1.summary, e2.summary)

    t1 = np.array(ifqig.history['theta']).squeeze()
    t2 = np.array(trlg.history['theta']).squeeze()
    assert np.allclose(t1, t2), "{}, {}".format(t1, t2)

    for v1, v2 in zip(ifqig.history['rho'], trlg.history['rho']):
        for sv1, sv2 in zip(v1, v2):
            assert np.allclose(sv1, sv2), "{}, {}".format(sv1, sv2)
