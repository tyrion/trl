import os

os.environ.setdefault('KERAS_BACKEND', 'theano')

import pytest
import theano
import theano.tensor as T
import numpy as np
from scipy import optimize

from ifqi import envs

from trl import algorithms, regressor, utils
from trl.experiment import Experiment


def bellmanop(rho, theta):
    return theta.dot(rho)


def lqr_reg(s, a, theta):
    b = theta[0]
    k = theta[1]
    return - b * b * s * a - 0.5 * k * a * a - 0.4 * k * s * s


def np_norm(x, p=2):
    return np.max(x ** 2) if p == np.inf else np.mean(x ** p) ** (1 / p)


def empirical_bop(e: Experiment, rho, theta0, norm_value=2, incremental=False):
    s = e.dataset.state
    a = e.dataset.action
    r = e.dataset.reward
    snext = e.dataset.next_state
    n = len(s)

    theta0_0 = theta0[0]
    if incremental:
        theta1_0 = theta0_0 + bellmanop(rho, theta0)[0]
    else:
        theta1_0 = bellmanop(rho, theta0)[0]
    qnop = lqr_reg(s, a, theta1_0)
    bop = -np.ones(n) * np.inf
    for i in range(n):
        for action in e.actions:
            qv = lqr_reg(snext[i], action, theta0_0)
            if qv > bop[i]:
                bop[i] = qv
    v = qnop - r - e.gamma * bop
    return np_norm(v, norm_value)


class LBPO(regressor.Regressor):
    def __init__(self, rho):
        self.rho = theano.shared(value=rho, name='rho')
        self.theta = T.matrix()
        self.outputs = [T.dot(self.theta, self.rho)]
        self.inputs = [self.theta]

        # do not update rho
        self.trainable_weights = [self.rho]
        #self.trainable_weights = []
        self.predict = theano.function(self.inputs, self.outputs[0])

    def model(self, theta):
        return bellmanop(self.rho, theta)

    def predict(self, x):
        pass

    def fit(self, x, y):
        raise NotImplementedError


class CurveFitQRegressor(regressor.Regressor):
    def __init__(self, params):
        self._params = p = theano.shared(params, 'params', allow_downcast=True)
        self.sa = sa = T.dmatrix('sa')
        self.s, self.a = sa[:, 0], sa[:, 1]
        self.j = self.model(sa, p)
        self.inputs = [sa]
        self.outputs = [self.j]
        self.predict = theano.function([sa], self.j)

    @property
    def trainable_weights(self):
        return [self._params]

    @property
    def params(self):
        return self._params.get_value()

    @params.setter
    def params(self, params):
        self._params.set_value(params)

    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params - 0.0001)

    def Q(self, sa, b, k):
        return self.model(sa, [b, k])

    def model(self, sa, theta):
        s = sa[:, 0]
        a = sa[:, 1]
        return lqr_reg(s, a, theta)

    def predict(self, x):
        pass


dataset_arr = np.array([
    [1., 0., 2., -1., 0., 0.],
    [2., 3., 3., -5., 0., 0.],
    [3., 4., 4., +0., 0., 0.],
], dtype=float)

dataset_rec = np.rec.array(dataset_arr.ravel(), copy=False, dtype=[
    ('state', float), ('action', float), ('reward', float),
    ('next_state', float), ('absorbing', float), ('done', float)])


rho0 = np.array([[1., 2.], [0., 3.]], dtype='f')
theta0 = np.array([[2., 0.2]], dtype='f')


@pytest.fixture(params=[(2, False), (2, True)])
def experiment(request):
    norm_value, incremental = request.param

    e = Experiment(
        env_name='LQG1D-v0',
        training_episodes=10,
        algorithm_class=algorithms.GradPBO,
        algorithm_config={
            'bo': LBPO(rho0),
            'K': 1,
            'optimizer': 'adam',
            'batch_size': 10,
            'norm_value': norm_value,
            'update_index': 10,
            'update_steps': None,
            'incremental': incremental,
            'independent': False,
        },
        np_seed=None,
        env_seed=None,
    )

    # e.dataset = e.get_dataset()
    e.dataset = dataset_rec
    e.q = CurveFitQRegressor(theta0[0])
    e.seed(1)
    e.algorithm = e.get_algorithm()
    return e


def test_bellman_error(experiment):
    e = experiment
    pbo = e.algorithm

    # dataset_arr = utils.rec_to_array(e.dataset)
    err0 = pbo.train_f(dataset_arr, theta0)
    err1 = empirical_bop(e, rho0, theta0, pbo.norm_value, pbo.incremental)

    assert np.allclose(err0, err1, 0.1)


def test_grad(experiment):
    e = experiment
    pbo = e.algorithm

    t_grad = T.grad(pbo.t_output, pbo.bo.trainable_weights)
    grad = theano.function(pbo.t_input, t_grad)
    r0 = grad(dataset_arr, theta0)

    eps = np.sqrt(np.finfo(float).eps)
    f = lambda x: empirical_bop(e, x, theta0)
    r1 = optimize.approx_fprime(rho0, f, eps)

    assert np.allclose(r0, r1)
