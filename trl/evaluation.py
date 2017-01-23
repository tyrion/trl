import logging
import time

import numpy as np
from ifqi.envs import get_space_info

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


def allocate_dataset(env, n=1):
    state_dim, action_dim, reward_dim = get_space_info(env)

    dataset = np.zeros(n, [
        ('state', float, state_dim),
        ('action', float, action_dim),
        ('reward', float, reward_dim),
        ('next_state', float, state_dim),
        ('absorbing', float, 1),
        ('done', float, 1),
    ])
    dataset = np.rec.array(dataset, copy=False)
    return dataset


def _get_reset(env, initial_states=None):
    if initial_states is None:
        return env.reset

    try:
        env.reset(initial_states[0])
    except TypeError:
        logging.warning('The env does not support setting the state. '
                        'Trying with `env.state = state`')
        env.reset_ = env.reset
        env.reset = lambda self, s: (setattr(self, 'state', s), s)[1]

    i = -1
    def reset():
        nonlocal i
        i += 1
        return env.reset(initial_states[i])

    return reset


def interact(env, n=1, horizon=100, policy=None, collect=False,
                     metrics=(), render=False, initial_states=None):
    if policy is None:
        policy = RandomPolicy(env)

    fps = env.metadata.get('video.frames_per_second') or 100
    dataset = allocate_dataset(env, horizon * (n if collect else 1))

    info = None
    if metrics:
        info = np.recarray((n,),
                [('time', int)] + [(m.__name__, float) for m in metrics])

    reset = _get_reset(env, initial_states)

    i = 0
    for e in range(n):
        state = reset()
        logger.info('Episode %d starting in %s', e, state)
        episode = dataset[i:i+horizon] if collect else dataset

        for t in range(horizon):
            if render:
                env.render()
                time.sleep(1 / fps)

            action = policy.draw_action(state.reshape(1, -1))
            next_state, reward, done, _ = env.step(action)
            episode[t] = (state, action, reward, next_state, 0, done)
            logger.debug(episode[t])

            if done:
                episode = episode[:t+1]
                break
            state = next_state
        episode[t].absorbing = True

        t += 1
        i += t

        logger.info('Episode %d finished in %d steps.', e, t)
        if metrics:
            m = tuple(m(episode) for m in metrics)
            info[e] = (t,) + m
            logger.info('Metrics: %s', m)

    return dataset[:i] if collect else None, info



def discounted(gamma=0.9):
    f = lambda e: sum(e.reward * np.power(gamma, np.arange(len(e))))
    f.__name__ = 'discounted'
    return f

def average(e):
    return e.reward.mean()

