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




class Experiment_:
    gamma = 0.9
    horizon = 100

    training_episodes = 100
    training_iterations = 50
    evaluation_episodes = 20

    use_action_regressor = False

    algorithm_config = {}
    algorithm_class = None

    timeit = 0
    render = False

    IGNORE_CONFIG = ('gamma', 'horizon', 'algorithm_config')

    def __init__(self, **config):
        self.config = config

        for key, value in config.items():
            if key not in self.IGNORE_CONFIG and hasattr(self.__class__, key):
                setattr(self, key, value)

    def run(self):
        self.log_config()

        self.dataset = self.get_dataset()
        #self.save_dataset(self.dataset_save_path)

        self.training_time = None
        self.trace = None
        self.summary = None

        self.seed(3)
        self.q = self.get_q()

        if self.training_iterations <= 0:
            logger.info('Skipping training.')
        else:
            self.seed(1)
            self.algorithm_config = self.get_algorithm_config()
            self.algorithm = self.get_algorithm()
            logger.info('Training algorithm (iterations: %d)',
                        self.training_iterations)
            fn = self.benchmark if self.timeit else self.train
            fn()

        # using 'not' instead of '<= 0' because it could be an array
        if not self.evaluation_episodes:
            logger.info('Skipping evaluation.')
        else:
            self.seed(2)
            self.evaluate()

        self.save()
        return (self.training_time, self.summary)

    def get_q(self):
        q = self.config['q']
        get_q = lambda i=0: q(self.state_dim + i, 1) if callable(q) else q
        return (ActionRegressor(get_q(), self.actions)
                if self.use_action_regressor else get_q(self.action_dim))

    def get_algorithm_config(self):
        config = copy.deepcopy(self.__class__.algorithm_config)
        config.update(config.get('algorithm_config', {}))
        return config

    def benchmark(self):
        t = timeit.repeat('self.train()', number=1,
                          repeat=self.timeit, globals=locals())
        self.training_time = min(t)
        logger.info('%d iterations, best of %d: %fs',
                    self.training_iterations, self.timeit, self.training_time)

    def save(self):
        # TODO save initial_states
        self.save_q(self.q_save_path)
        if self.evaluation_episodes > 0:
            self.save_trace(self.trace_save_path)
        if self.save_path is not None:
            if self.training_iterations > 0:
                self.algorithm.save(self.save_path)
            self.save_config('{}.json'.format(self.save_path))

    def save_q(self, path):
        if path is not None:
            attrs = {'time': self.training_time} if self.timeit else None
            regressor.save_regressor(self.q, path, 'q', attrs)

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



def action(method):
    sig = inspect.signature(method)
    name = '{}_{{}}'.format(method.__name__)

    def wrapper(self, *args, **kwargs):
        return self._action(method, sig, *args, **kwargs)

    return wrapper



import inspect
import collections




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
            seed = int.from_bytes(os.urandom(max_bytes * 2), 'big')
        self._seed = seed
        npy_seed, env_seed = divmod(seed, 2 ** (8 * max_bytes))
        self.npy_seed = npy_seed
        self.env_seed = env_seed
        self.stage = 0

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
                 collect=None, metrics=(), render=False, stage=None):
        policy = policy(self)
        i = evaluation.Interaction(self.env, episodes, self.horizon, policy, collect, metrics, render)
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

    def evaluate(self, metrics=None, **kwargs):
        kwargs['collect'] = False
        if metrics is None:
            metrics = evaluation.average, evaluation.discounted(self.gamma)
        return self.interact(metrics=metrics, **kwargs)

    def train(self, *, q, algorithm_class, dataset=None, iterations=100,
              output=None, stage=None, algorithm_config=None):
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

        self.policy = evaluation.QPolicy(algo.q, self.actions)

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
