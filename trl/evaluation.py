import logging
import time

import numpy as np

from . import utils


logger = logging.getLogger(__name__)


class QPolicy:
    def __init__(self, q, actions):
        self.q = q
        self.actions = actions

    def draw_action(self, states, absorbing=False, evaluation=False):
        v = self.q(utils.make_grid(states, self.actions))
        return self.actions[v.argmax()]


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def draw_action(self, *args):
        return self.env.action_space.sample()


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
                logging.warning('The env does not support setting the state. '
                                'Trying with `env.state = state`')
                _reset = lambda s: (setattr(self, 'state', s), s)[1]
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
            ('state_i', float, self.state_dim),
            ('state_f', float, self.state_dim),
            ('time', int)
        ] + ([('metrics', float, len(m))] if m else []))

    def allocate_dataset(self):
        n = self.horizon * (self.n if self.collect else 1)
        self.dataset = np.recarray((n,), [
            ('state', float, self.state_dim),
            ('action', float, self.action_dim),
            ('reward', float),
            ('next_state', float, self.state_dim),
            ('absorbing', float),
            ('done', float),
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
                next_state, reward, done, _ = self.env.step(action)
                episode[t] = (state, action, reward, next_state, 0, done)
                logger.debug(episode[t])

                if done:
                    episode = episode[:t+1]
                    break
                state = next_state
            episode[t].absorbing = True

            t += 1
            i += t

            trace.state_f = state
            trace.time = t
            if self.metrics:
                trace.metrics = [m(episode) for m in self.metrics]

            logger.info('Episode %3d: %s', e, trace)


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

