import concurrent.futures
import copy
import hashlib
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


class Experiment:
    env_name = None

    horizon = 100
    gamma = 0.9
    initial_states = None

    training_episodes = 100
    training_iterations = 50
    evaluation_episodes = 20
    budget = None

    use_action_regressor = False

    algorithm_config = {}
    algorithm_class = None

    np_seed = None
    env_seed = None

    timeit = 0
    render = False

    save_path = None
    dataset_load_path = None
    dataset_save_path = None
    q_load_path = None
    q_save_path = None
    trace_save_path = None


    IGNORE_CONFIG = ('gamma', 'horizon', 'algorithm_config',
                     'training_episodes', 'evaluation_episodes')

    def __init__(self, **config):
        assert config.get('initial_states') is None
        self.config = config
        self.env_name = config.pop('env_name', self.env_name)
        self.env = self.get_env()
        self.env_spec = self.env.unwrapped.spec
        self.gamma = self.get_gamma()
        self.horizon = self.get_horizon()

        self.save_path = path = config.pop('save_path', self.save_path)
        if self.save_path is not None:
            for path in ('dataset_save_path', 'q_save_path', 'trace_save_path'):
                if getattr(self, path, None) is None:
                    setattr(self, path, self.save_path)

        for key, value in config.items():
            if key not in self.IGNORE_CONFIG and hasattr(self.__class__, key):
                setattr(self, key, value)

        self.state_dim = utils.get_space_dim(self.env.observation_space)
        self.action_dim = utils.get_space_dim(self.env.action_space)
        self.actions = self.get_actions()

        self.np_seed = init_seed(self.np_seed)
        self.env_seed = init_seed(self.env_seed)

    def get_env(self):
        return gym.make(self.env_name)

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

    def get_actions(self):
        return utils.discretize_space(self.env.action_space)

    def run(self):
        self.log_config()
        self.dataset = self.get_dataset()
        self.save_dataset(self.dataset_save_path)

        self.training_time = None
        self.trace = None
        self.summary = None

        self.seed(3)
        if self.use_action_regressor:
            self.input_dim = self.state_dim
            self.q = regressor.ActionRegressor(self.get_q(), self.actions)
        else:
            self.input_dim = self.state_dim + self.action_dim
            self.q = self.get_q()

        if self.training_iterations <= 0:
            logger.info('Skipping training.')
        else:
            self.seed(1)
            self.algorithm_config = self.get_algorithm_config()
            self.algorithm = self.get_algorithm()
            logger.info('Training algorithm (iterations: %d, budget: %s)',
                        self.training_iterations, self.budget)
            fn = self.benchmark if self.timeit else self.train
            fn()

        self.seed(2)
        self.evaluation_episodes = self.get_evaluation_episodes()
        self.evaluate()

        self.save()
        return (self.training_time, self.summary)

    def log_config(self):
        logger.info('Initialized env %s', self.env_name)
        logger.info('observation space: %s', self.env.observation_space)
        logger.info('action space: %s', self.env.action_space)
        logger.info('Random seeds (np, env): %s %s', self.np_seed, self.env_seed)
        logger.info('Discretized actions (%d): %s', len(self.actions),
            np.array2string(self.actions, max_line_width=np.inf))
        logger.info('Gamma: %f', self.gamma)


    def seed(self, stage):
        np_seed = get_seed(self.np_seed, stage)
        env_seed = int.from_bytes(get_seed(self.env_seed, stage), 'big')

        logger.info('Stage %d seeds (np, env): %d %d',
                stage, int.from_bytes(np_seed, 'big'), env_seed)

        np_seed = [int.from_bytes(bytes(b), 'big')
                    for b in zip(*[iter(np_seed)]*4)]
        np.random.seed(np_seed)
        self.env.seed(env_seed)

    def get_dataset(self):
        try:
            path = self.config['dataset_load_path']
        except KeyError:
            self.seed(0)
            self.training_episodes = self.get_training_episodes()
            interaction = evaluation.Interact(self.env, self.training_episodes,
                                              self.horizon, collect=True)
            logger.info('Collecting training data (episodes: %d, horizon: %d)',
                        interaction.n, self.horizon)
            interaction.interact()
            self.env.reset()
            return interaction.dataset
        else:
            return utils.load_dataset(path)

    def get_q(self):
        return regressor.load_regressor(self.q_load_path, name='q')

    def get_training_episodes(self):
        return self.config.get('training_episodes',
                               self.__class__.training_episodes)

    def get_algorithm_config(self):
        config = copy.deepcopy(self.__class__.algorithm_config)
        config.update(self.config.get('algorithm_config', {}))
        return config

    def get_algorithm(self):
        return self.algorithm_class(self, **self.algorithm_config)

    def get_evaluation_episodes(self):
        try:
            return self.config['evaluation_episodes']
        except KeyError:
            return getattr(self.env, 'initial_states',
                           self.__class__.evaluation_episodes)

    def train(self):
        self.algorithm.run(self.training_iterations, self.budget)

    def benchmark(self):
        t = timeit.repeat('self.train()', number=1,
                          repeat=self.timeit, globals=locals())
        self.training_time = min(t)
        logger.info('%d iterations, best of %d: %fs',
                    self.training_iterations, self.timeit, self.training_time)

    def evaluate(self):
        self.policy = evaluation.QPolicy(self.q, self.actions)

        metrics = [evaluation.average, evaluation.discounted(self.gamma)]
        interaction = evaluation.Interact(
            self.env, self.evaluation_episodes, self.horizon, self.policy,
            self.render, metrics)

        if interaction.n <= 0:
            logger.info('Skipping evaluation.')
            return

        logger.info('Evaluating algorithm (episodes: %d)', interaction.n)
        interaction.interact()
        self.trace = t = interaction.trace
        self.summary = np.concatenate((t.time.mean(keepdims=True),
                                       t.metrics.mean(0)))
        logger.info('Summary avg (time: %f, avgJ: %f, discountedJ: %f)',
                    *self.summary)
        self.save_trace(self.trace_save_path)

    def save(self):
        # TODO save initial_states
        self.save_q(self.q_save_path)
        if self.save_path is not None:
            if self.training_iterations > 0:
                self.algorithm.save(self.save_path)
            self.save_config('{}.json'.format(self.save_path))

    def save_dataset(self, path):
        if path is not None:
            utils.save_dataset(self.dataset, path)

    def save_q(self, path):
        if path is not None:
            attrs = {'time': self.training_time} if self.timeit else None
            regressor.save_regressor(self.q, path, 'q', attrs)

    def save_trace(self, path):
        if path is not None:
            utils.save_dataset(self.trace, path, 'trace')
            utils.save_dataset(self.summary, path, 'summary')

    _CONFIG_KEYS = ['env_name', 'horizon', 'gamma',
        'training_iterations', 'budget',
        'use_action_regressor', 'np_seed', 'env_seed',
        'timeit', 'render', 'dataset_load_path', 'dataset_save_path',
        'q_load_path', 'q_save_path', 'save_path']

    def get_config(self):
        config = {k: getattr(self, k) for k in self._CONFIG_KEYS}
        # FIXME handle algorithm_config and initial_states
        a = self.algorithm_class
        config['algorithm_class'] = ':'.join([a.__module__, a.__name__])
        return config

    def save_config(self, path):
        with open(path, 'w') as fp:
            json.dump(self.get_config(), fp, sort_keys=True, indent=4)


    @classmethod
    def run_ith(cls, i, **config):
        cls._setup(i, **config)

        config = {k: (v.format(i=i)
                  if k.endswith('path') and isinstance(v, str) else v)
                  for k, v in config.items()}
        e = cls(**config)
        return e.run()

    @classmethod
    def run_many(cls, n, run=None, workers=None, **config):
        if run is None:
            run = cls.run_ith
        return cls.run_iter(range(n), run, workers, **config)

    @classmethod
    def run_iter(cls, iter, run, workers=None, **config):
        logger = logging.getLogger('trl')
        lvl = logger.level
        logger.setLevel(logging.ERROR)

        with concurrent.futures.ProcessPoolExecutor(workers) as executor:
            start_time = time.time()
            futures = {executor.submit(run, x, **config): i
                       for i, x in enumerate(iter)}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results[i] = r = future.result()
                except Exception as exc:
                    logging.info('%d generated an exception.' % i, exc_info=1)
                else:
                    logging.info('Experiment %s completed: %s', i, r)
            t = time.time() - start_time
            logging.info('Finished in %f (avg: %f)', t, t / len(futures))

        logger.setLevel(lvl)
        return results


    @classmethod
    def _setup(cls, i, **config):
        logging.disable(logging.INFO)
