import logging
import warnings

import numpy as np
from pybrain.optimization import ExactNES

from .. import regressor, utils
from ..experiment import Experiment


# pybrain is giving a lot of deprecation warnings
warnings.filterwarnings('ignore', module='pybrain')

logger = logging.getLogger('trl.algorithms')


class Algorithm:
    def __init__(self, experiment: Experiment):
        self.dataset = experiment.dataset
        self.actions = experiment.actions
        self.gamma = experiment.gamma
        self.q = experiment.q
        self.experiment = experiment
        self.S1A = utils.make_grid(self.dataset.next_state, self.actions)
        self.SA = utils.rec_to_array(self.dataset[['state', 'action']])

    def max_q(self):
        n = len(self.dataset)
        n_actions = len(self.actions)

        y = self.q(self.S1A).reshape((n, n_actions))

        # XXX why? we could avoid to compute the absorbing states altogether.
        y = y * (1 - self.dataset.absorbing)[:, np.newaxis]
        #amax = np.argmax(y, axis=1)
        return y.max(axis=1)

    def first_step(self, budget=None):
        self.step(i=0, budget=budget)

    def run(self, n=10, budget=None):
        logger.info('Iteration 0')
        self.first_step(budget)
        for i in range(1, n):
            logger.info('Iteration %d', i)
            self.step(i, budget)

    def save(self, path):
        """Save algorithm state to file"""
        pass


class FQI(Algorithm):

    def first_step(self, budget=None):
        # XXX I think first step like this is ok only if self.q is 0
        self.q.fit(self.SA, self.dataset.reward)

    def step(self, i=0, budget=None):
        y = self.dataset.reward + self.gamma * self.max_q()
        self.q.fit(self.SA, y)
        #log(i, y, self.params)


class PBO(Algorithm):

    def __init__(self, experiment, bo, K=1, incremental=False):
        super().__init__(experiment)
        self.bo = bo
        self.K = K
        self.incremental = incremental

    def loss(self, omega):
        with self.bo.save_params(omega), self.q.save_params():
            loss = 0
            for k in range(self.K):
                q0 = self.max_q()

                t = self.bo.predict_one(self.q.params)
                self.q.params = (self.q.params + t) if self.incremental else t

                q1 = self.q(self.SA)
                v = q1 - self.dataset.reward - self.gamma * q0
                loss += utils.norm(v, 2)
        logger.debug('loss: %7d | q: %s', loss, self.q.params)
        #np.array2string(omega, max_line_width=np.inf))
        return loss

    def save(self, path):
        regressor.save_regressor(self.bo, path, 'bo')


class NESPBO(PBO):

    def __init__(self, experiment, bo, K=1, incremental=False, batch_size=10,
                 learning_rate=0.1, **nes_args):
        super().__init__(experiment, bo, K)
        self.incremental = incremental
        nes_args.setdefault('importanceMixing', False)
        self.best_params = self.bo.params
        self.optimizer = ExactNES(self.loss, self.best_params, minimize=True,
                                  batchSize=batch_size,
                                  learningRate=learning_rate, **nes_args)

    def loss(self, omega):
        loss = super().loss(omega)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = omega
        return loss

    def step(self, i=0, budget=None):
        self.best_loss = np.inf
        _, g_loss = self.optimizer.learn(budget)
        self.bo.params = self.best_params
        tnext = self.bo.predict_one(self.q.params)
        self.q.params = (self.q.params + tnext) if self.incremental else tnext
        logger.info('Global best: %f | Local best: %f', g_loss, self.best_loss)
