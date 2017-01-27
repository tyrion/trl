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

    def __init__(self, **config):
        self.config = config
        self.env_name = config.pop('env_name', self.env_name)
        self.env = self.get_env()

        for key in ['gamma', 'horizon', 'initial_states']:
            config.setdefault(key, getattr(self.env, key))

        self.algorithm_config = copy.deepcopy(self.algorithm_config)
        self.algorithm_config.update(config.pop('algorithm_config', {}))

        self.save_path = path = config.pop('save_path', self.save_path)
        if self.save_path is not None:
            for path in ('dataset_save_path', 'q_save_path', 'trace_save_path'):
                if getattr(self, path, None) is None:
                    setattr(self, path, self.save_path)

        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        e = self.evaluation_episodes
        i = len(self.initial_states)
        if self.initial_states is not None and (e is None or e > i):
            self.evaluation_episodes = i

        self.state_dim = utils.get_space_dim(self.env.observation_space)
        self.action_dim = utils.get_space_dim(self.env.action_space)
        self.actions = self.get_actions()

        self.np_seed = init_seed(self.np_seed)
        self.env_seed = init_seed(self.env_seed)

    def get_env(self):
        return gym.make(self.env_name)

    def get_actions(self):
        return utils.discretize_space(self.env.action_space)

    def run(self):
        self.log_config()
        self.dataset = self.get_dataset()
        self.training_time = None
        self.trace = None
        self.summary = None

        if self.use_action_regressor:
            self.input_dim = self.state_dim
            self.q = regressor.ActionRegressor(self.get_q(), self.actions)
        else:
            self.input_dim = self.state_dim + self.action_dim
            self.q = self.get_q()

        self.algorithm_config = self.get_algorithm_config()
        self.algorithm = self.get_algorithm()

        if self.training_iterations <= 0:
            logger.info('Skipping training.')
        else:
            logger.info('Training algorithm (iterations: %d, budget: %s)',
                        self.training_iterations, self.budget)
            self.seed(1)
            fn = self.benchmark if self.timeit else self.train
            fn()

        if self.evaluation_episodes <= 0:
            logger.info('Skipping evaluation.')
        else:
            logger.info('Evaluating algorithm (episodes: %d)',
                        self.evaluation_episodes)
            self.seed(2)
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
            logger.info('Collecting training data (episodes: %d, horizon: %d)',
                        self.training_episodes, self.horizon)
            self.seed(0)

            interaction = evaluation.Interact(self.env, self.training_episodes,
                                              self.horizon, collect=True)
            interaction.interact()
            self.env.reset()
            return interaction.dataset
        else:
            return utils.load_dataset(path)

    def get_q(self):
        return regressor.load_regressor(self.q_load_path, name='q')

    def get_algorithm_config(self):
        # XXX not saving to self, be careful when saving the Experiment.
        return self.algorithm_config

    def get_algorithm(self):
        return self.algorithm_class(self, **self.algorithm_config)

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

        n = self.initial_states
        n = n if n is not None else self.evaluation_episodes

        metrics = [evaluation.average, evaluation.discounted(self.gamma)]
        interaction = evaluation.Interact(self.env, n, self.horizon,
                                          self.policy, self.render, metrics)
        interaction.interact()
        self.trace = t = interaction.trace
        self.summary = np.concatenate((t.time.mean(keepdims=True),
                                       t.metrics.mean(0)))
        logger.info('Summary avg (time: %f, avgJ: %f, discountedJ: %f)',
                    *self.summary)

    def save(self):
        # TODO save initial_states
        self.save_dataset(self.dataset_save_path)
        self.save_q(self.q_save_path)
        if self.evaluation_episodes > 0:
            self.save_trace(self.trace_save_path)
        if self.save_path is not None:
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

    _CONFIG_KEYS = ['env_name', 'horizon', 'gamma', 'training_episodes',
        'training_iterations', 'evaluation_episodes', 'budget',
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
    def run_many(cls, n, run=None, **config):
        if run is None:
            run = cls.run_ith
        return cls.run_iter(range(n), run, **config)

    @classmethod
    def run_iter(cls, iter, run, **config):
        logger = logging.getLogger('trl')
        lvl = logger.level
        logger.setLevel(logging.ERROR)

        with concurrent.futures.ProcessPoolExecutor() as executor:
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
