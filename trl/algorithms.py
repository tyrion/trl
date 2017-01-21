import logging

import numpy as np
from numpy.linalg import norm
from pybrain.optimization import ExactNES

from . import utils


logger = logging.getLogger(__name__)


class Algorithm:
    def __init__(self, dataset, actions, gamma=1):
        self.dataset = dataset
        self.actions = actions
        self.gamma = 1
        self.S1A = utils.make_grid(dataset.next_state, actions)
        self.SA = utils.rec_to_array(dataset[['state', 'action']])

    def max_q(self, q):
        n = len(self.dataset)
        n_actions = len(self.actions)

        y = q(self.S1A).reshape((n, n_actions))

        # XXX why? we could avoid to compute the absorbing states altogether.
        y = y * (1 - self.dataset.absorbing)[:, np.newaxis]
        amax = np.argmax(y, axis=1)
        return y.max(axis=1)

    def first_step(self, budget=None):
        self.step(i=0, budget=budget)

    def run(self, n=10, budget=None):
        logger.info('Iteration 0')
        self.first_step(budget)
        for i in range(1, n):
            logger.info('Iteration %d', i)
            self.step(i, budget)


class FQI(Algorithm):

    def __init__(self, dataset, actions, q, gamma=1):
        super().__init__(dataset, actions)
        self.q = q
        self.gamma = 1

    def first_step(self, budget=None):
        # XXX I think first step like this is ok only if self.q is 0
        self.q.fit(self.SA, self.dataset.reward)

    def step(self, i=0, budget=None):
        y = self.dataset.reward + self.gamma * self.max_q(self.q)
        self.q.fit(self.SA, y)
        #log(i, y, self.params)


class PBO(Algorithm):

    def __init__(self, dataset, actions, q, bo, gamma=1, K=1):
        super().__init__(dataset, actions, gamma)
        self.bo = bo
        self.q = q
        self.K = K

    def loss(self, omega):
        with self.bo.save_params(omega), self.q.save_params():
            loss = 0
            for k in range(self.K):
                q0 = self.max_q(self.q)

                self.q.params = self.bo(self.q.params)
                q1 = self.q(self.SA)
                loss += norm(q1 - self.dataset.reward + self.gamma * q1, 2)
        logger.debug('loss: %7d | q: %s', loss, self.q.params)
        #np.array2string(omega, max_line_width=np.inf))
        return loss


## ipotesi per il logging
# 1. passarre oggetto custom come msg, con __str__ e usare un handler
# 2.


class LoggingNES(ExactNES):
    prev = None

    def _notify(self):
        b = self.bestEvaluation
        if b != self.prev:
            n = self.numLearningSteps
            logger.info('%4s loss: %d', n, b)
            self.prev = b


class NESPBO(PBO):

    def __init__(self, dataset, actions, q, bo, gamma=1, K=1, **nes_args):
        super().__init__(dataset, actions, q, bo, gamma, K)
        self.optimizer = LoggingNES(self.loss, self.bo.params, minimize=True,
                                    importanceMixing=False, **nes_args)

    def step(self, i=0, budget=None):
        params, loss = self.optimizer.learn(budget)
        self.bo.params = params
        self.q.params = self.bo(self.q.params)
        #print(loss, delta)
        #log(i, params, loss, self.q.params)
