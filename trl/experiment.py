import logging
import timeit

import h5py
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from trl import evaluation, regressor, utils


logger = logging.getLogger(__name__)


class Experiment:
    env_name = None

    horizon = None
    gamma = None
    budget = None

    np_seed = None
    env_seed = None

    training_episodes = 100
    training_iterations = 50
    evaluation_episodes = None
    initial_states = None

    algorithm = None

    timeit = 0
    render = False

    load_path = None
    save_path = None
    dataset_load_path = None
    dataset_save_path = None
    q_load_path = None
    q_save_path = None


    def __init__(self, initopts, opts):
        self._set_options(initopts)
        self._set_options(opts)

    def _set_options(self, opts):
        for k, v in opts.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def make(cls, **initopts):

        def experiment(**opts):
            self = cls(initopts, opts)
            return self.run()

        return experiment


    def run(self):
        self.env = self.load_env()
        self.np_seed, self.env_seed = self.seed(self.np_seed, self.env_seed)

        logger.info('Initialized env %s', self.env_name)
        logger.info('observation space: %s', self.env.observation_space)
        logger.info('action space: %s', self.env.action_space)
        logger.info('Random seeds (np, env): %s %s', self.np_seed, self.env_seed)

        self.actions = self.get_actions()
        logger.info('Discretized actions (%d): %s', len(self.actions),
            np.array2string(self.actions, max_line_width=np.inf))

        self.state_dim = utils.get_space_dim(self.env.observation_space)
        self.action_dim = utils.get_space_dim(self.env.action_space)

        self.input_dim = self.state_dim
        if not self.use_action_regressor:
            self.input_dim += self.action_dim

        self.horizon = self.get_horizon()
        self.gamma = self.get_gamma()

        self.initial_states = i = self.get_initial_states()
        if self.evaluation_episodes is None:
            self.evaluation_episodes = len(i) if i is not None else 10

        logger.info('Gamma: %f', self.gamma)

        self.dataset, self.q = self.load(self.load_path)
        self.algorithm = self.get_algorithm()

        if self.training_iterations <= 0:
            logger.info('Skipping training.')
        else:
            logger.info('Training algorithm (iterations: %d, budget: %s)',
                        self.training_iterations, self.budget)
            fn = self.benchmark if self.timeit else self.train
            fn(self.training_iterations, self.budget)

        if self.evaluation_episodes <= 0:
            logger.info('Skipping evaluation.')
        else:
            logger.info('Evaluating algorithm (episodes: %d)',
                        self.evaluation_episodes)
            self.evaluate()

        self.save(self.save_path)
        return self

    def load_env(self):
        return gym.make(self.env_name)

    def seed(self, np_seed=None, env_seed=None):
        # seeding np.random (for pybrain)
        np_seed = seeding._seed(np_seed)
        np.random.seed(
            seeding._int_list_from_bigint(seeding.hash_seed(np_seed)))
        env_seeds = self.env.seed(env_seed)
        env_seed = env_seeds[0] if env_seeds else None
        return np_seed, env_seed

    def get_actions(self):
        return utils.discretize_space(self.env.action_space)

    def get_horizon(self):
        return self.horizon or getattr(self.env, 'horizon', 100)

    def get_gamma(self):
        return self.gamma or getattr(self.env, 'gamma', 0.9)

    def get_initial_states(self):
        # FIXME allow random
        return self.initial_states or getattr(self.env, 'initial_states', None)

    def get_algorithm(self):
        return self.algorithm

    def load(self, path):
        if path is not None:
            self.dataset_load_path = path
            self.q_load_path = path

        dataset = self.get_dataset(self.dataset_load_path)
        logger.info('Collected %d samples', len(dataset))
        q = self.get_q(self.q_load_path)
        return dataset, q

    def get_dataset(self, path=None):
        if path is not None:
            dataset = utils.load_dataset(path)
        else:
            logger.info('Collecting training data (episodes: %d, horizon: %d)',
                        self.training_episodes, self.horizon)
            dataset, _ = evaluation.interact(self.env, self.training_episodes,
                                             self.horizon, collect=True)
            self.env.reset()
        return dataset

    def get_q(self, path):
        r = regressor.load_regressor(path)
        if self.use_action_regressor:
            return regressor.ActionRegressor(r, self.actions)
        return r

    @property
    def use_action_regressor(self):
        return not isinstance(self.env.action_space, spaces.Box)

    def save(self, path):
        if path is not None:
            self.dataset_save_path = path
            self.q_save_path = path

        self.save_dataset(self.dataset_save_path, self.dataset)
        self.save_q(self.q_save_path, self.q)


    def save_dataset(self, path, dataset):
        if path is not None:
            utils.save_dataset(dataset, path)

    def save_q(self, path, q):
        if path is not None:
            regressor.save_regressor(q, path)

    def train(self, iterations, budget=None):
        return self.algorithm.run(iterations, budget)

    def benchmark(self, repeat, iterations, budget=None):
        t = timeit.repeat('self.train(iterations, budget)', number=1,
                          repeat=repeat, globals=locals())
        logger.info('%d iterations, best of %d: %fs',
                self.training_iterations, repeat, min(t))

    def evaluate(self):
        self.policy = evaluation.QPolicy(self.q, self.actions)
        d, info = evaluation.interact(self.env, self.evaluation_episodes,
            self.horizon, self.policy, render=self.render,
            initial_states=self.initial_states,
            metrics=[evaluation.average, evaluation.discounted(self.gamma)])

        logger.info('Summary avg (time: %f, avgJ: %f, discountedJ: %f)',
                     info.time.mean(), info.average.mean(),
                     info.discounted.mean())

    def save_regressor(self, path, regressor):
        regressor.save_regressor(path, regressor)
