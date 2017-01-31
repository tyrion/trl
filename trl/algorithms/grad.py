import logging
import time

import numpy as np
import theano
from theano import tensor as T
from keras.engine.training import slice_X, batch_shuffle, make_batches
from keras import optimizers

from .base import Algorithm
from .. import regressor, utils


ZERO = np.array(0, dtype='float64')
logger = logging.getLogger('trl.algorithms')


def get_shape(dim):
    return (-1, dim)[:dim]


def apply(fn, n, i):
    res = []
    for _ in range(n):
        i = fn(i)
        res.append(i)
    return res


def t_pnorm(x, p=2):
    """p-norm as defined in IFQI"""
    return T.max(x ** 2) if p == np.inf else T.mean(x ** p) ** (1 / p)


def t_make_grid(x, y):
    nx = x.shape[0]
    ny = y.shape[0]

    x = x.reshape((nx, -1))
    y = y.reshape((ny, -1))

    x = T.tile(x, (1, ny)).reshape((-1, x.shape[1]))
    y = T.tile(y, (nx, 1))
    return T.concatenate((x, y), axis=-1)


class GradientAlgorithm(Algorithm):

    def __init__(self, experiment, optimizer='adam', batch_size=10,
                 norm_value=2, update_index=1):
        super().__init__(experiment)
        assert len(self.q.inputs) == 1
        assert len(self.q.outputs) == 1
        self.optimizer = optimizers.get(optimizer)
        self.batch_size = batch_size
        self.norm_value = norm_value
        self.update_index = update_index

        self.t_actions = T.constant(self.actions)
        self.batches = make_batches(len(self.dataset), batch_size)

        self.indices = np.arange(len(self.dataset))
        self.x = []

    def compile(self, trainable_weights, inputs, loss):
        start_time = time.time()
        o = self.optimizer
        updates = o.get_updates(trainable_weights, {}, loss)
        self.train_f = theano.function(inputs, [loss], updates=updates,
                                       name='train', allow_input_downcast=True)
        logger.info('Compiled train_f in %fs', time.time() - start_time)

    def step(self, i=0, budget=None):
        np.random.shuffle(self.indices)
        i = i * self.batch_size

        for start, end in self.batches:
            batch = slice_X(self.data, self.indices[start:end])
            i += 1
            self.train_f(*(batch + self.x))

            if self.update_index > 0 and i % self.update_index == 0:
                self.update_inputs()


class GradFQI(GradientAlgorithm):

    def __init__(self, experiment, optimizer='adam', batch_size=10,
                 norm_value=2, update_index=1):
        super().__init__(experiment, optimizer, batch_size, norm_value, update_index)

        self.t_y = t_y = T.dvector('y')
        loss = t_pnorm(self.q.outputs[0] - t_y, norm_value)
        self.compile(self.q.trainable_weights, self.q.inputs + [t_y], loss)

        self.update_inputs()

    def update_inputs(self):
        #logging.debug('Theta %s', self.q.params)
        self.data = [self.SA, self.dataset.reward + self.gamma * self.max_q()]


class GradPBO(GradientAlgorithm):

    def __init__(self, experiment, bo, K=1, optimizer='adam', batch_size=10,
                 norm_value=2, update_index=1, update_steps=None,
                 incremental=False, independent=False):
        super().__init__(experiment, optimizer, batch_size, norm_value, update_index)
        assert len(bo.inputs) == 1
        assert len(bo.outputs) == 1

        self.bo = bo
        self.K = K
        self.incremental = incremental
        self.update_steps = K if update_steps is None else update_steps
        self.independent = independent

        # Theano variables (prefixed with 't_')
        self.t_dataset = t_d = T.dmatrix('dataset')

        s_dim = experiment.state_dim
        a_dim = experiment.action_dim
        r_idx = s_dim + a_dim
        n_idx = r_idx + 1
        a_idx = n_idx + s_dim

        # Ensuring that Theano variables have the right shapes.
        #self.t_s = t_d[:, 0:s_dim].reshape(get_shape(s_dim))
        #self.t_a = t_d[:, s_dim:r_idx].reshape(get_shape(a_dim))
        self.t_sa = t_d[:, :r_idx].reshape(get_shape(r_idx))
        self.t_r = t_d[:, r_idx]
        self.t_s_next = t_d[:, n_idx:a_idx].reshape(get_shape(s_dim))
        self.t_absorbing = t_d[:, a_idx]

        self.t_s1a = t_make_grid(self.t_s_next, self.t_actions)

        t_theta0 = bo.inputs[0]
        t_thetas = [t_theta0]

        if not independent:
            loss = self.k_loss(t_theta0)
        else:
            loss = self.loss(t_theta0)[1]
            for i in range(1, K):
                # XXX shouldn't this be 'dmatrix'?
                theta_i = T.fmatrix('theta_{}'.format(i))
                t_thetas.append(theta_i)
                loss += self.loss(theta_i)[1]
            assert len(t_thetas) == K

        self.compile(bo.trainable_weights, [t_d] + t_thetas, loss)

        # Variables needed during execution
        self.data = [utils.rec_to_array(self.dataset)]
        self.theta0 = self.q.params.reshape(1,-1)
        self.apply_bo = (lambda t: t + bo(t)) if incremental else bo
        self.update_thetas = (lambda t: [t]) if not independent else \
                             (lambda t: apply(self.apply_bo, K, t))
        self.x = self.update_thetas(self.theta0)

    # s, a version instead of sa
    # def max_q(self, s, theta):
    #     q_values, _ = theano.scan(lambda a, s, t: self.q.model(s, a, t),
    #                               [self.t_actions],
    #                               non_sequences=[s, theta])
    #     return T.max(q_values)
    # And then:
    #     maxq, _ = theano.scan(self.max_q, [self.t_s_next],
    #                           non_sequences=[theta0_0])

    def max_q(self, theta):
        n = self.t_dataset.shape[0]
        n_actions = len(self.actions)

        y = self.q.model(self.t_s1a, theta).reshape((n, n_actions))
        y = y * (1 - self.t_absorbing)[:, np.newaxis]

        return y.max(axis=1)

    def loss(self, theta0, loss=ZERO):
        theta1 = self.bo.model(theta0)
        if self.incremental:
            theta1 += theta0

        theta0_0 = theta0[0]
        theta1_0 = theta1[0]

        qpbo = self.q.model(self.t_sa, theta1_0)

        maxq = self.max_q(theta0_0)
        v = qpbo - self.t_r - self.gamma * maxq
        return theta1, loss + t_pnorm(v, self.norm_value)

    def k_loss(self, theta0):
        (_, loss), _ = theano.scan(self.loss, outputs_info=[theta0, ZERO],
                                   n_steps=self.K)
        return loss[-1]

    def update_inputs(self):
        for _ in range(self.update_steps):
            self.theta0 = self.apply_bo(self.theta0)
        self.x = self.update_thetas(self.theta0)


    def run(self, n=10, budget=None):
        super().run(n, budget)
        # logging.info('Learned theta: %s', self.theta0[0])
        self.q.params = self.theta0[0]

    def save(self, path):
        regressor.save_regressor(self.bo, path, 'bo')
