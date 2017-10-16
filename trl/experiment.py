import collections
import contextlib
import copy
import hashlib
import inspect
import json
import logging
import os
import time
import timeit

import h5py
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from trl import evaluation, regressor, utils


logger = logging.getLogger(__name__)


def init_seed(seed, max_bytes=8):
    if seed is None:
        return int.from_bytes(os.urandom(max_bytes), 'big')
    return int(seed) % 2 ** (8 * max_bytes)


def get_seed(seed, stage, max_bytes=8):
    seed = (seed + stage) % 2 ** (8 * max_bytes)
    seed = seed.to_bytes(max_bytes, 'big')
    return hashlib.sha512(seed).digest()[-max_bytes:]



def action(method):
    sig = inspect.signature(method)
    name = '{}_{{}}'.format(method.__name__)

    def wrapper(self, *args, **kwargs):
        return self._action(method, sig, *args, **kwargs)

    return wrapper


def _get(keys, dicts):
    *keys, last_key = keys
    *dicts, last_dict = dicts
    for key, dict in zip(keys, dicts):
        try:
            return dict[key]
        except KeyError:
            pass
    return last_dict[last_key]


class Experiment:
    algorithm = None
    interaction = None
    policy = None
    summary = None

    gamma = 0.9
    horizon = 100

    def __init__(self, env_spec, **config):
        self.env_spec = env_spec
        self.env = gym.make(env_spec.id)
        self.config = config
        self.actions = self.get_actions()
        self.state_dim = utils.get_space_dim(self.env.observation_space)
        self.action_dim = utils.get_space_dim(self.env.action_space)
        self.gamma = self.get_gamma()
        self.horizon = self.get_horizon()

        self.init_seed()

    def _config_env_class(self, key):
        try:
            return self.config[key]
        except KeyError:
            return getattr(self.env, key, getattr(self.__class__, key))

    def get_gamma(self):
        return self._config_env_class('gamma')

    def get_horizon(self):
        return (self.env_spec.timestep_limit or
                self._config_env_class('horizon'))

    def init_seed(self, max_bytes=8):
        seed = self.config.get('seed')
        if seed is None:
            seed_bytes = os.urandom(max_bytes * 2)
            seed = int.from_bytes(seed_bytes, 'big')
        else:
            seed = seed % 256 ** (max_bytes * 2)
            seed_bytes =  seed.to_bytes(max_bytes * 2, 'big')
        self._seed = seed
        npy_seed, env_seed = divmod(seed, 256 ** max_bytes)
        self.npy_seed = npy_seed
        self.env_seed = env_seed
        self.stage = 0

        save_seed = self.config.get('save_seed')
        if save_seed:
            utils.save_dataset(np.array(seed_bytes), save_seed, 'seed')

    def seed(self, stage=None):
        stage = stage if stage is not None else self.stage

        npy_seed = get_seed(self.npy_seed, stage)
        env_seed = get_seed(self.env_seed, stage)

        npy_seed_int = int.from_bytes(npy_seed, 'big')
        env_seed_int = int.from_bytes(env_seed, 'big')

        logger.info('Stage %d seeds (npy, env): %d %d',
                    stage, npy_seed_int, env_seed_int)

        npy_seed = [int.from_bytes(bytes(b), 'big')
                    for b in zip(*[iter(npy_seed)]*4)]
        np.random.seed(npy_seed)
        self.env.seed(env_seed_int)
        self.stage = stage + 1

    def get_actions(self):
        return utils.discretize_space(self.env.action_space)

    def log_config(self):
        logger.info('Initialized env %s', self.env_spec.id)
        logger.info('observation space: %s', self.env.observation_space)
        logger.info('action space: %s', self.env.action_space)
        logger.info('Random seed: %d', self._seed)
        logger.info('Discretized actions (%d): %s', len(self.actions),
            np.array2string(self.actions, max_line_width=np.inf))
        logger.info('Gamma: %f', self.gamma)


    def interact(self, *, policy=lambda e: None, episodes=100, output=None,
                 collect=None, metrics=(), render=False, stage=None,
                 log_level=None):
        with self.setup_logging(log_level):
            policy = policy(self)
            i = evaluation.Interaction(self.env, episodes, self.horizon,
                                       policy, collect, metrics, render)
            self.seed(stage)
            i.run()

            self.interaction = i

            if metrics:
                t = i.trace
                s = np.concatenate((t.time.mean(keepdims=True), t.metrics.mean(0)))
                self.summary = s
                logger.info('Summary avg (time, *metrics): %s', s)

            if output:
                if collect:
                    utils.save_dataset(i.dataset, output)
                utils.save_dataset(i.trace, output, 'trace')

            return i

    def collect(self, **kwargs):
        kwargs['collect'] = True
        return self.interact(**kwargs)

    def evaluate(self, *, metrics=None, **kwargs):
        kwargs['collect'] = False
        if metrics is None:
            metrics = evaluation.average, evaluation.discounted(self.gamma)
        return self.interact(metrics=metrics, **kwargs)


    def benchmark(self):
        t = timeit.repeat('self.train()', number=1,
                          repeat=self.timeit, globals=locals())
        self.training_time = min(t)
        logger.info('%d iterations, best of %d: %fs',
                    self.training_iterations, self.timeit, self.training_time)

    def train(self, *, q, algorithm_class, dataset=None, iterations=100,
              output=None, stage=None, log_level=None, **algorithm_config):
        with self.setup_logging(log_level):
            stage_a, stage_b = stage or (stage, stage)
            if algorithm_config is None:
                algorithm_config = {}

            if dataset is None:
                if self.interaction is not None and self.interaction.collect:
                    dataset = self.interaction.dataset
                else:
                    raise click.UsageError('Missing dataset.')

            self.seed(stage_a)
            algorithm_config['dataset'] = dataset
            algorithm_config['actions'] = self.actions
            algorithm_config['gamma'] = self.gamma
            algorithm_config['horizon'] = self.horizon
            algorithm_config['q'] = q(self.state_dim + self.action_dim, 1)

            self.seed(stage_b)
            algo = algorithm_class(**algorithm_config)
            algo.run(iterations)

            if output:
                regressor.save_regressor(algo.q, output, 'q')
                algo.save(output)

            self.algorithm = algo
            self.policy = evaluation.QPolicy(algo.q, self.actions)

    @contextlib.contextmanager
    def setup_logging(self, level, logger='trl'):
        if level is not None:
            logger = logging.getLogger(logger)
            initial_level = logger.level
            logger.setLevel(level)
            yield
            logger.setLevel(initial_level)
        else:
            yield

    def run(self):
        self.collect()
        self.train()
        self.evaluate()


    # _get_default('collect_ep', 'interact_ep')
    def _get_default(self, keys, params={}):
        pass

    def _apply_defaults(self, ba, default_keys):
        arguments = ba.arguments
        new_arguments = []
        config = (arguments, self.config, vars(self.__class__))

        for name, param in ba._signature.parameters.items():
            try:
                key = default_key.format(name)
                value = _get([name, key, key], config)
            except KeyError:
                if param.kind is inspect._VAR_POSITIONAL:
                    value = ()
                elif param.kind is inspect._VAR_KEYWORD:
                    value = {}
                else:
                    continue
            new_arguments.append((name, value))
        ba.arguments = collections.OrderedDict(new_arguments)
        return ba

    def _action(self, method, sig, args, kwargs, name=None):
        ba = sig.bind_partial(self, *args, **kwargs)
        ba = self._apply_defaults(ba, name)
        return method(*ba.args, **ba.kwargs)
