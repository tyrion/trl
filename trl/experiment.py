import logging
import timeit


import gym
import numpy as np
from gym.utils import seeding

from trl import evaluation, utils


logger = logging.getLogger(__name__)


class Experiment:

    def __init__(self, env, algorithm, q=None, horizon: int = None,
                 gamma: float = None, budget: int = None,
                 training_episodes: int = 100,
                 training_iterations: int = 50,
                 evaluation_episodes: int = None,
                 initial_states=None, np_seed=None, env_seed=None,
                 render=True,
                 **algorithm_options):
        self.env = self.get_env(env)
        self.np_seed, self.env_seed = self.set_seeds(np_seed, env_seed)

        logger.info('Initialized env %s', env)
        logger.info('observation space: %s', self.env.observation_space)
        logger.info('action space: %s', self.env.action_space)

        self.training_episodes = training_episodes
        self.training_iterations = training_iterations
        self.evaluation_episodes = evaluation_episodes
        self.budget = budget
        self.render = render

        self.actions = self.get_actions()
        logger.info('Discretized actions (%d): %s', len(self.actions),
            np.array2string(self.actions, max_line_width=np.inf))

        self.state_dim = utils.get_space_dim(self.env.observation_space)
        self.action_dim = utils.get_space_dim(self.env.action_space)
        self.input_dim = self.state_dim + self.action_dim

        self.horizon = self.get_horizon(horizon)
        self.gamma = self.get_gamma(gamma)
        self.initial_states = i = self.get_initial_states(initial_states)
        if self.evaluation_episodes is None:
            self.evaluation_episodes = len(i) if i is not None else 10

        logger.info('Gamma: %f', self.gamma)

        self.dataset = self.get_dataset()
        self.q = self.get_q(q)
        self.algorithm = self.get_algorithm(algorithm, **algorithm_options)

    def get_env(self, env):
        return gym.make(env)

    def set_seeds(self, np_seed=None, env_seed=None):
        # seeding np.random (for pybrain)
        np_seed = seeding._seed(np_seed)
        np.random.seed(
            seeding._int_list_from_bigint(seeding.hash_seed(np_seed)))
        env_seeds = self.env.seed(env_seed)
        env_seed = env_seeds[0] if env_seeds else None
        logger.info('Random seeds (np, env): %s %s', np_seed, env_seed)
        return np_seed, env_seed

    def get_actions(self):
        return utils.discretize_space(self.env.action_space)

    def get_horizon(self, horizon=None):
        return horizon or getattr(self.env, 'horizon', 100)

    def get_gamma(self, gamma=None):
        return gamma or getattr(self.env, 'gamma', 0.9)

    def get_initial_states(self, initial_states=None):
        # FIXME allow random
        return initial_states or getattr(self.env, 'initial_states', None)

    def get_dataset(self):
        logger.info('Collecting training data (episodes: %d, horizon: %d)',
                    self.training_episodes, self.horizon)
        dataset, _ = evaluation.interact(self.env, self.training_episodes,
                                         self.horizon, collect=True)
        logger.info('Collected %d samples', len(dataset))
        self.env.reset()
        return dataset

    def get_q(self, q):
        return q

    def get_algorithm(self, algorithm, **kwargs):
        return algorithm(self, **kwargs)

    def train(self):
        logger.info('Training algorithm (iterations: %d, budget: %s)',
                    self.training_iterations, self.budget)
        return self.algorithm.run(self.training_iterations, self.budget)

    def benchmark(self, repeat):
        t = timeit.repeat('self.train()', number=1, repeat=repeat,
                          globals=locals())
        logger.info('%d iterations, best of %d: %fs',
                self.training_iterations, repeat, min(t))

    def evaluate(self):
        logger.info('Evaluating algorithm (episodes: %d)',
                    self.evaluation_episodes)
        policy = evaluation.QPolicy(self.q, self.actions)
        evaluation.interact(self.env, self.evaluation_episodes, self.horizon,
                            policy, render=self.render,
                            initial_states=self.initial_states,
                            metrics=[evaluation.average,
                                     evaluation.discounted(self.gamma)])
