import numpy as np
import theano

from ifqi.algorithms import fqi
from ifqi.algorithms.pbo import pbo, gradpbo
from ifqi.models.regressor import Regressor

from .base import Algorithm
from .. import utils


class FQI(Algorithm):
    def __init__(self, q, dataset, actions, gamma, horizon):
        super().__init__(q, dataset, actions, gamma, horizon)

        self.fqi = fqi.FQI(
            estimator=self.q,
            state_dim=dataset.state.ndim,
            action_dim=dataset.action.ndim,
            discrete_actions=self.actions,
            gamma=self.gamma,
            horizon=self.horizon,
            verbose=False)

        self.sast = utils.rec_to_array(self.dataset[
                                           ['state', 'action', 'next_state', 'absorbing']])
        dtype = self.dataset.reward.dtype
        self.r = self.dataset.reward.view(dtype, np.ndarray)

    def first_step(self):
        return self.fqi.partial_fit(self.sast, self.r)

    def step(self, i=0):
        self.fqi.partial_fit()


class PBO(Algorithm):
    def __init__(self, q, dataset, actions, gamma, horizon, bo,
                 incremental=False, batch_size=10, learning_rate=0.1):
        super().__init__(q, dataset, actions, gamma, horizon)

        r = Regressor(object)
        r._regressor = self.q
        bo = bo(self.q) if callable(bo) else bo

        self.pbo = pbo.PBO(
            estimator=r,
            estimator_rho=bo._model,
            state_dim=dataset.state.ndim,
            action_dim=dataset.action.ndim,
            discrete_actions=self.actions,
            gamma=self.gamma,
            learning_steps=10,
            batch_size=batch_size,
            learning_rate=learning_rate,
            incremental=incremental,
            verbose=False)

        self.sast = utils.rec_to_array(self.dataset[
                                           ['state', 'action', 'next_state', 'absorbing']])
        dtype = self.dataset.reward.dtype
        self.r = self.dataset.reward.view(dtype, np.ndarray)

    def run(self, n=10):
        self.pbo._learning_steps = n
        self.pbo.fit(self.sast, self.r)


class GradPBO(Algorithm):
    class Q:
        def __init__(self, regressor):
            self.regressor = regressor

        def model(self, s, a, omega):
            sa = theano.tensor.stack((s, a), 1)
            q = self.regressor.model([sa], [omega[0]])
            return q[0].ravel()


    class BO:
        def __init__(self, regressor):
            self.regressor = regressor

        def model(self, inputs):
            return self.regressor.model([inputs])[0]

        def __getattr__(self, key):
            return getattr(self.regressor, key)

    def __init__(self, q, dataset, actions, gamma, horizon, bo, K=1,
                 optimizer='adam', batch_size=10, norm_value=2, update_index=1,
                 update_steps=None, incremental=False, independent=False):
        super().__init__(q, dataset, actions, gamma, horizon)

        self.Q = self.Q(self.q)
        bo = bo(self.q) if callable(bo) else bo
        self.pbo = gradpbo.GradPBO(
            bellman_model=self.BO(bo),
            q_model=self.Q,
            state_dim=dataset.state.ndim,
            action_dim=dataset.action.ndim,
            discrete_actions=self.actions,
            gamma=self.gamma,

            steps_ahead=K,
            optimizer=optimizer,
            incremental=incremental,
            update_theta_every=update_index,
            steps_per_theta_update=update_steps,
            norm_value=norm_value,
            independent=independent,
            verbose=0)
        self.batch_size = batch_size

    def run(self, n=10):
        d = self.dataset
        theta0 = self.q.params.reshape(1, -1)
        self.history = self.pbo.fit(d.state, d.action, d.next_state, d.reward,
                                    d.absorbing, theta0, self.batch_size, n)
        thetaf = self.pbo.learned_theta_value[0]
        self.q.params = thetaf

    def apply_bo(self, theta):
        return self.pbo.apply_bo(theta)
