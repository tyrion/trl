import logging
import time

import numpy as np
import theano

from . import utils
from ifqi.utils.spaces.sampler import space_sampler


logger = logging.getLogger(__name__)
floatX = theano.config.floatX


class QPolicy:
    def __init__(self, q, actions):
        self.q = q
        self.actions = actions

    def draw_action(self, states, absorbing=False, evaluation=False):
        v = self.q(utils.make_grid(states, self.actions))
        return self.actions[v.argmax()]


class RandomPolicy:
    def __init__(self, env):
        self.sampler = space_sampler(env.action_space)

    def draw_action(self, *args):
        return self.sampler()


class Interact:

    def __init__(self, env, n=1, horizon=None, policy=None, collect=True,
                 metrics=(), render=False):
        if isinstance(n, int):
            self.n = n
            self.initial_states = None
            self.reset = lambda: env.reset()
        else:
            self.n = len(n)
            self.initial_states = n
            try:
                env.reset(n[0])
            except TypeError:
                unw = env.unwrapped
                key = 'state'
                if not hasattr(unw, key):
                    raise

                logging.warning('The env does not support setting the '
                                'state. Trying with `env.state = state`')
                _reset = lambda s: (env.reset(), setattr(unw, key, s), s)[2]
            else:
                _reset = env.reset
            self.reset = lambda: _reset(n[self.e])

        self.env = env
        self.state_dim = utils.get_space_dim(self.env.observation_space)
        self.action_dim = utils.get_space_dim(self.env.action_space)

        self.horizon = horizon or getattr(env, 'horizon', 100)
        self.policy = RandomPolicy(env) if policy is None else policy
        self.collect = collect
        self.metrics = metrics
        self.render = render

        self.allocate_dataset()
        self.allocate_trace()

    def allocate_trace(self):
        m = self.metrics
        self.trace = np.recarray((self.n,), [
            ('state_i', floatX, self.state_dim),
            ('state_f', floatX, self.state_dim),
            ('time', int)
        ] + ([('metrics', floatX, len(m))] if m else []))

    def allocate_dataset(self):
        n = self.horizon * (self.n if self.collect else 1)
        self.dataset = np.recarray((n,), [
            ('state', floatX, self.state_dim),
            ('action', floatX, self.action_dim),
            ('reward', floatX),
            ('next_state', floatX, self.state_dim),
            ('absorbing', floatX),
            ('done', floatX),
        ])

    def __iter__(self):
        return iter(self.trace)

    def interact(self):
        i = 0
        for e in range(self.n):
            self.e = e
            trace = self.trace[e]
            trace.state_i = state = self.reset()
            episode = (self.dataset[i:i+self.horizon]
                        if self.collect else self.dataset)

            for t in range(self.horizon):
                if self.render:
                    self.env.render()
                    time.sleep(1 / fps)

                action = self.policy.draw_action(state.reshape(1, -1))
                next_state, reward, absorbing, _ = self.env.step(action)
                episode[t] = (state, action, reward, next_state, absorbing, 0)
                logger.debug(episode[t])

                state = next_state
                if absorbing:
                    episode = episode[:t+1]
                    break
            episode[t].done = True

            t += 1
            i += t

            trace.state_f = state
            trace.time = t
            if self.metrics:
                trace.metrics = [m(episode) for m in self.metrics]

            logger.info('Episode %3d: %s', e, trace)
        self.dataset = self.dataset[:i]


def interact(env, n=1, horizon=100, policy=None, collect=False,
                     metrics=(), render=False):
    i = Interact(env, n, horizon, policy, collect, metrics, render)
    i.interact()
    return i.dataset, i.trace


def discounted(gamma=0.9):
    f = lambda e: sum(e.reward * np.power(gamma, np.arange(len(e))))
    f.__name__ = 'discounted'
    return f

def average(e):
    return e.reward.mean()

