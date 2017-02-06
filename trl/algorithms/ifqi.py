import numpy as np
import theano

from ifqi.algorithms import fqi
from ifqi.algorithms.pbo import pbo, gradpbo
from ifqi.models.regressor import Regressor

from .base import Algorithm
from .. import utils


class FQI(Algorithm):
    def __init__(self, experiment):
        super().__init__(experiment)

        self.fqi = fqi.FQI(
            estimator=self.q,
            state_dim=experiment.state_dim,
            action_dim=experiment.action_dim,
            discrete_actions=self.actions,
            gamma=self.gamma,
            horizon=experiment.horizon,
            verbose=False)

        self.sast = utils.rec_to_array(self.dataset[
                                           ['state', 'action', 'next_state', 'absorbing']])
        dtype = self.dataset.reward.dtype
        self.r = self.dataset.reward.view(dtype, np.ndarray)

    def first_step(self, budget=None):
        return self.fqi.partial_fit(self.sast, self.r)

    def step(self, i=0, budget=None):
        self.fqi.partial_fit()


class PBO(Algorithm):
    def __init__(self, experiment, bo, incremental=False, batch_size=10,
                 learning_rate=0.1):
        super().__init__(experiment)

        r = Regressor(object)
        r._regressor = self.q

        self.pbo = pbo.PBO(
            estimator=r,
            estimator_rho=bo._model,
            state_dim=experiment.state_dim,
            action_dim=experiment.action_dim,
            discrete_actions=self.actions,
            gamma=self.gamma,
            learning_steps=experiment.training_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            incremental=incremental,
            verbose=False)

        self.sast = utils.rec_to_array(self.dataset[
                                           ['state', 'action', 'next_state', 'absorbing']])
        dtype = self.dataset.reward.dtype
        self.r = self.dataset.reward.view(dtype, np.ndarray)

    def run(self, n=10, budget=None):
        self.pbo.fit(self.sast, self.r)


class GradPBO(Algorithm):
    class Q:
        def __init__(self, regressor):
            self.regressor = regressor

        def model(self, s, a, omega):
            sa = theano.tensor.stack((s, a), 1)
            q = self.regressor.model(sa, omega[0])
            return q.ravel()

    def __init__(self, experiment, bo, K=1, optimizer='adam', batch_size=10,
                 norm_value=2, update_index=1, update_steps=None,
                 incremental=False, independent=False):
        super().__init__(experiment)

        self.q = self.Q(self.q)
        self.pbo = gradpbo.GradPBO(
            bellman_model=bo,
            q_model=self.q,
            state_dim=experiment.state_dim,
            action_dim=experiment.action_dim,
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

    def run(self, n=10, budget=None):
        d = self.dataset
        theta0 = self.q.regressor.params.reshape(1, -1)
        self.history = self.pbo.fit(d.state, d.action, d.next_state, d.reward, theta0,
                                    self.batch_size, n, verbose=0)
        thetaf = self.pbo.learned_theta_value[0]
        self.q.regressor.params = thetaf

    def apply_bo(self, theta):
        return self.pbo.apply_bo(theta)
