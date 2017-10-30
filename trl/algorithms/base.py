import abc
import logging
import warnings

import click
import numpy as np
from pybrain.optimization import ExactNES

from trl import cli, evaluation, regressor, utils


# pybrain is giving a lot of deprecation warnings
warnings.filterwarnings('ignore', module='pybrain')

logger = logging.getLogger('trl.algorithms')


class AlgorithmMeta(abc.ABCMeta):
    registry = {}

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        if (any(b for b in bases if isinstance(b, mcls)) and
            not cls.__abstractmethods__):
            mcls.registry[cls.cli_name] = cls

        return cls

    @property
    def cli_name(cls):
        return cls.__name__.lower()


def configure_train_stages(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    if len(value) == 1:
        return (value[0], value[0] + 1)

    if len(value) == 2:
        return value[:2]

    raise click.UsageError("Cannot specify more than two stages.")


class Algorithm(metaclass=AlgorithmMeta):

    def __init__(self, q, dataset, actions, gamma, horizon):
        self.dataset = dataset
        self.actions = actions
        self.gamma = gamma
        self.horizon = horizon
        self.q = q
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

    def first_step(self):
        self.step(i=0)

    def run(self, n=10):
        self.create_history()
        logger.info('Iteration 0')
        self.first_step()
        for i in range(1, n):
            logger.info('Iteration %d', i)
            self.step(i)

    def save(self, path):
        """Save algorithm state to file"""
        pass

    def create_history(self):
        pass

    def update_history(self):
        pass

    # XXX add --timeit
    _LOGGING_OPT = cli.LoggingOption('trl.algorithms')
    cli_params = [
        # FIXME I think it should be cli.Regressor('q')
        click.Argument(('q',), type=cli.Regressor()),
        click.Option(('-i', '--iterations'), default=100),
        click.Option(('-d', '--dataset'), type=cli.DATASET),
        click.Option(('-o', '--output'), type=cli.PATH,
                     default=cli.default_output),
        click.Option(('-s', '--stage'), metavar='N', type=int, multiple=True,
                     callback=configure_train_stages),
        click.Option(('--ar', 'use_action_regressor'), default=False,
                     is_flag=True),
        _LOGGING_OPT
    ] + _LOGGING_OPT.options
    cli_kwargs = {}

    @classmethod
    def make_cli(cls):
        cb = cli.processor(cls.cli_callback)
        return click.Command(cls.cli_name, callback=cb,
                             params=cls.cli_params, **cls.cli_kwargs)

    @classmethod
    def cli_callback(cls, exp, **config):
        return exp.train(algorithm_class=cls, **config)


class FQI(Algorithm):

    def first_step(self):
        # XXX I think first step like this is ok only if self.q is 0
        self.q.fit(self.SA, self.dataset.reward)

    def step(self, i=0):
        y = self.dataset.reward + self.gamma * self.max_q()
        self.q.fit(self.SA, y)
        #log(i, y, self.params)


class PBO(Algorithm):
    cli_params = Algorithm.cli_params + [
        click.Argument(('bo',), type=cli.BO_REGRESSOR),
        click.Option(('-k', 'K'), default=1),
        click.Option(('--norm', 'norm_value'), type=float, default=2),
        click.Option(('--update-index',), default=1),
        click.Option(('--update-steps',), type=int),
        click.Option(('--inc/--no-inc', 'incremental'), is_flag=True,
                     default=False)
    ]

    def __init__(self, q, dataset, actions, gamma, horizon, bo, K=1, norm_value=2,
                 update_index=1, update_steps=None, incremental=False):
        super().__init__(q, dataset, actions, gamma, horizon)
        self.bo = bo(self.q) if callable(bo) else bo
        self.K = K
        self.norm_value = norm_value
        self.incremental = incremental
        self.update_index = update_index
        self.update_steps = K if update_steps is None else update_steps

    def loss(self, omega):
        with self.bo.save_params(omega), self.q.save_params():
            loss = 0
            for k in range(self.K):
                q0 = self.max_q()

                t = self.bo.predict_one(self.q.params)
                self.q.params = (self.q.params + t) if self.incremental else t

                q1 = self.q(self.SA)
                v = q1 - self.dataset.reward - self.gamma * q0
                loss += utils.norm(v, self.norm_value)
        logger.debug('loss: %7d | q: %s', loss, self.q.params)
        #np.array2string(omega, max_line_width=np.inf))
        return loss

    def save(self, path):
        regressor.save_regressor(self.bo, path, 'bo')


class NESPBO(PBO):
    cli_params = PBO.cli_params + [
        click.Option(('-B', '--budget'), metavar='N', type=int),
        click.Option(('-b', '--batch', 'batch_size'), default=10),
        click.Option(('-r', '--learning-rate'), default=0.1),
    ]

    def __init__(self, q, dataset, actions, gamma, horizon, bo, K=1, norm_value=2,
                 update_index=1, update_steps=None, incremental=False,
                 budget=None, batch_size=10, learning_rate=0.1, **nes_args):
        super().__init__(q, dataset, actions, gamma, horizon, bo, K, norm_value,
                         update_index, update_steps, incremental)
        nes_args.setdefault('importanceMixing', False)
        self.best_params = self.bo.params
        self.optimizer = ExactNES(self.loss, self.best_params, minimize=True,
                                  batchSize=batch_size,
                                  learningRate=learning_rate, **nes_args)
        self.budget = budget

    def loss(self, omega):
        loss = super().loss(omega)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = omega
        return loss

    def step(self, i=0):
        self.best_loss = np.inf
        _, g_loss = self.optimizer.learn(self.budget)
        self.bo.params = self.best_params
        logger.info('Global best: %f | Local best: %f', g_loss, self.best_loss)

        if self.update_index > 0 and i % self.update_index == 0:
            for _ in range(self.update_steps):
                tnext = self.bo.predict_one(self.q.params)
                self.q.params = (self.q.params + tnext)\
                    if self.incremental else tnext


class WeightedFQI(FQI):
    def __init__(self, *args, reinit=False, weight_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_steps = weight_steps

    # XXX workaround to specify weight_steps from CLI
    def run(self, n=10, budget=1):
        self.weight_steps = int(
            budget if self.weight_steps is None else self.weight_steps)
        logger.info('WeightedFQI w/ weight_steps=%d', self.weight_steps)
        return super().run(n, budget)

    def get_weights(self, y):
        _, idx, counts = np.unique(y, return_inverse=True, return_counts=True)
        w = (len(y) / (len(counts) * counts))[idx]
        return w

    def first_step(self, budget=None):
        y = self.dataset.reward
        w = self.get_weights(y) if self.weight_steps > 0 else None
        self.q.fit(self.SA, y, sample_weight=w)

    def step(self, i=0, budget=None):
        y = self.dataset.reward + self.gamma * self.max_q()
        w = self.get_weights(y) if self.weight_steps > i else None
        self.q.fit(self.SA, y, sample_weight=w)
        #log(i, y, self.params)
