import logging
import time

import numpy as np
import theano
from theano import tensor as T
from keras.engine.training import slice_X, batch_shuffle, make_batches
from keras import optimizers

from .base import Algorithm
from .. import regressor, utils

ZERO = np.array(0, dtype=theano.config.floatX)
logger = logging.getLogger('trl.algorithms')


def get_shape(dim):
    return (-1, dim)[:dim]


def apply(fn, n, i):
    res = []
    for _ in range(n):
        i = fn(i)
        res.append(i)
    return res


def t_make_grid(x, y):
    nx = x.shape[0]
    ny = y.shape[0]

    x = x.reshape((nx, -1))
    y = y.reshape((ny, -1))

    x = T.tile(x, (1, ny)).reshape((-1, x.shape[1]))
    y = T.tile(y, (nx, 1))
    return T.concatenate((x, y), axis=-1)


from theano.compile.debugmode import DebugMode


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
                                       name='train', allow_input_downcast=True,
                                       )#mode=DebugMode(check_isfinite=False))
        logger.info('Compiled train_f in %fs', time.time() - start_time)

    def step(self, i=0, budget=None):
        np.random.shuffle(self.indices)
        i = i * len(self.batches)

        for start, end in self.batches:
            self.update_history()
            batch = slice_X(self.data, self.indices[start:end])
            i += 1
            self.train_f(*(batch + self.x))

            if self.update_index > 0 and i % self.update_index == 0:
                self.update_inputs()


class GenGradFQI(GradientAlgorithm):
    def __init__(self, experiment, optimizer='adam', batch_size=10,
                 norm_value=2, update_index=1):
        super().__init__(experiment, optimizer, batch_size, norm_value, update_index)

        self.t_y = t_y = T.vector('y')
        loss = utils.norm(self.q.outputs[0] - t_y, norm_value)
        self.compile(self.q.trainable_weights, self.q.inputs + [t_y], loss)

        self.update_inputs()

    def update_inputs(self):
        # logging.debug('Theta %s', self.q.params)
        self.data = [self.SA, self.dataset.reward + self.gamma * self.max_q()]

    def create_history(self):
        self.history = {"theta":[]}

    def update_history(self):
        self.history["theta"].append(self.q.params)


def zip_sum(a, b):
    return [x + y for x, y in zip(a, b)]


class GradPBO(GradientAlgorithm):
    def __init__(self, experiment, bo, K=1, optimizer='adam', batch_size=10,
                 norm_value=2, update_index=1, update_steps=None,
                 incremental=False, independent=False):
        super().__init__(experiment, optimizer, batch_size, norm_value,
                         update_index)
        assert len(bo.inputs) == len(bo.outputs) == len(self.q.trainable_weights)

        self.bo = bo
        self.K = K
        self.incremental = incremental
        self.update_steps = K if update_steps is None else update_steps
        self.independent = independent

        # Theano variables (prefixed with 't_')
        self.t_dataset = t_d = T.matrix('dataset')

        s_dim = experiment.state_dim
        a_dim = experiment.action_dim
        r_idx = s_dim + a_dim
        n_idx = r_idx + 1
        a_idx = n_idx + s_dim

        # Ensuring that Theano variables have the right shapes.
        # self.t_s = t_d[:, 0:s_dim].reshape(get_shape(s_dim))
        # self.t_a = t_d[:, s_dim:r_idx].reshape(get_shape(a_dim))
        self.t_sa = t_d[:, :r_idx].reshape(get_shape(r_idx))
        self.t_r = t_d[:, r_idx]
        self.t_s_next = t_d[:, n_idx:a_idx].reshape(get_shape(s_dim))
        self.t_absorbing = t_d[:, a_idx]

        self.t_s1a = t_make_grid(self.t_s_next, self.t_actions)

        t_theta0 = bo.inputs
        assert isinstance(bo.inputs, list)
        #t_thetas = [t_theta0] if not isinstance(t_theta0, list) else t_theta0

        if not independent:
            loss = self.t_k_loss(t_theta0)
            #loss = self.t_loss(ZERO, *t_theta0)[0]
        else:
            raise NotImplementedError()
            loss = self.t_loss(ZERO, *t_theta0)[0]
            for i in range(1, K):
                # XXX shouldn't this be 'dmatrix'?
                theta_i = T.matrix('theta_{}'.format(i))
                t_thetas.append(theta_i)
                loss += self.t_loss(ZERO, *theta_i)[0]
            assert len(t_thetas) == K


        self.t_input = [t_d] + t_theta0
        self.t_output = loss
        self.compile(bo.trainable_weights, self.t_input, self.t_output)

        # Variables needed during execution
        self.data = [utils.rec_to_array(self.dataset)]
        self.theta0 = [w.get_value()[np.newaxis, ...] for w in
                       self.q.trainable_weights]
        # Keras does not return a list if len(output) is 1
        bo = (lambda x: [self.bo(x)]) if len(bo.outputs) == 1 else bo
        self.apply_bo = (lambda t: zip_sum(t, bo(t))) if incremental else bo
        self.update_thetas = (lambda t: t) if not independent else \
                             (lambda t: apply(self.apply_bo, K, t))
        self.x = self.update_thetas(self.theta0)

    def t_max_q(self, theta):
        n = self.t_dataset.shape[0]
        n_actions = len(self.actions)

        y = self.q.model([self.t_s1a], theta)[0].reshape((n, n_actions))
        y = y * (1 - self.t_absorbing)[:, np.newaxis]

        return y.max(axis=1)

    def t_loss(self, loss, *theta):
        maxq = self.t_max_q([t[0] for t in theta])

        tnext = self.bo.model(theta)
        theta = zip_sum(theta, tnext) if self.incremental else tnext

        qpbo = self.q.model([self.t_sa], [t[0] for t in theta])[0].ravel()
        v = qpbo - self.t_r - self.gamma * maxq
        return [loss + utils.norm(v, self.norm_value)] + theta

    def t_k_loss(self, theta0):
        (loss, *_), _ = theano.scan(self.t_loss, outputs_info=[ZERO] + theta0,
                                    n_steps=self.K)
        return loss[-1]

    def update_inputs(self):
        for _ in range(self.update_steps):
            self.theta0 = self.apply_bo(self.theta0)
        self.x = self.update_thetas(self.theta0)

    def run(self, n=10, budget=None):
        super().run(n, budget)
        # logging.info('Learned theta: %s', self.theta0[0])
        thetaf = self.theta0
        # for _ in range(100):
        #     thetaf = self.apply_bo(thetaf)
        # self.q.params = thetaf[0]
        # FIXME
        self.q.params = np.concatenate([w.ravel() for w in thetaf])

    def save(self, path):
        regressor.save_regressor(self.bo, path, 'bo')

    def create_history(self):
        self.history = {"theta":[], 'rho':[]}

    def update_history(self):
        self.history["theta"].append(self.x[0])
        self.history["rho"].append(self.bo._model.get_weights())
