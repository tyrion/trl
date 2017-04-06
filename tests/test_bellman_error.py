import pytest
import numpy as np
import numdifftools as nd
import theano
from theano import tensor as T

from ifqi import envs

from trl import algorithms, regressor, utils
from trl.experiment import Experiment

floatX = theano.config.floatX


def bellmanop(rho, theta):
    return theta.dot(rho)


def lqr_reg(s, a, theta):
    b = theta[0]
    k = theta[1]
    return - b * b * s * a - 0.5 * k * a * a - 0.4 * k * s * s


class LBPO(regressor.TheanoRegressor):
    def _model(self, theta, params):
        return bellmanop(params, theta)


class CurveFitQRegressor(regressor.TheanoRegressor):
    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params - 0.0001)

    def Q(self, sa, b, k):
        return self.model(sa, [b, k])

    def _model(self, sa, theta):
        s = sa[:, 0]
        a = sa[:, 1]
        return lqr_reg(s, a, theta)


dataset_arr = np.array([
    [1., 0., -1., 2., 0., 0.],
    [2., 3., -5., 3., 0., 0.],
    [3., 4., +0., 4., 0., 0.],
], dtype=floatX)

dataset_rec = np.rec.array(dataset_arr.ravel(), copy=False, dtype=[
    ('state', floatX), ('action', floatX), ('reward', floatX),
    ('next_state', floatX), ('absorbing', floatX), ('done', floatX)])

rho0 = np.array([[1., 2.], [0., 3.]], dtype=floatX)
theta0 = np.array([[2., 0.2]], dtype=floatX)


def build_bo(q):
    return LBPO(rho0)


params = {
    '1': (False, 1),
    '3': (False, 3),
    '1inc': (True, 1),
    '3inc': (True, 3),
}


@pytest.fixture(params=list(params.values()), ids=list(params.keys()))
def experiment(request):
    incremental, K = request.param
    config = {
        'q': CurveFitQRegressor(theta0[0]),
        'dataset': dataset_rec,
        'actions':  np.array([1, 2, 3], dtype=floatX).reshape(-1, 1),
        'gamma': 0.99,
        'horizon': 100,
        'bo': build_bo,
        'K': K,
        'norm_value': 2,
        'update_index': 10,
        'update_steps': None,
        'incremental': incremental,
    }

    epbo = algorithms.PBO(**config)
    grad = algorithms.GradPBO(**config)
    return epbo, grad


def test_bellman_error(experiment):
    epbo, grad = experiment

    err0 = epbo.loss(rho0)
    err1 = grad.train_f(dataset_arr, theta0)

    assert np.allclose(err0, err1)


def test_bellman_grad(experiment):
    epbo, gpbo = experiment

    t_grad = T.grad(gpbo.t_output, gpbo.bo.trainable_weights)
    grad = theano.function(gpbo.t_input, t_grad, name='grad')
    r0 = grad(dataset_arr, theta0)

    f = lambda x: epbo.loss(x.reshape(rho0.shape))
    dfun = nd.Gradient(f)
    r1 = dfun(rho0.ravel()).reshape(rho0.shape)

    assert np.allclose(r0, r1)
