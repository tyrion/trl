import numpy as np

from ifqi.algorithms import fqi
from ifqi.algorithms.pbo import pbo
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
        self.r = self.dataset.reward.view(float, np.ndarray)

    def first_step(self, budget=None):
        return self.fqi.partial_fit(self.sast, self.r)

    def step(self, i=0, budget=None):
        self.fqi.partial_fit()



class PBO(Algorithm):

    def __init__(self, experiment, bo):
        super().__init__(experiment)

        r = Regressor(object)
        r._regressor = self.q

        self.pbo = pbo.PBO(estimator=r,
            estimator_rho=bo._model,
            state_dim=experiment.state_dim,
            action_dim=experiment.action_dim,
            discrete_actions=self.actions,
            gamma=self.gamma,
            learning_steps=experiment.training_iterations,
            batch_size=10,
            learning_rate=0.1,
            incremental=False,
            verbose=False)

        self.sast = utils.rec_to_array(self.dataset[
            ['state', 'action', 'next_state', 'absorbing']])
        dtype = self.dataset.reward.dtype
        self.r = self.dataset.reward.view(dtype, np.ndarray)

    def run(self, n=10, budget=None):
        self.pbo.fit(self.sast, self.r)
