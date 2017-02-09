import logging
from contextlib import closing

from sklearn.utils.extmath import cartesian
import h5py
import numpy as np
import gym
import theano
from gym import spaces

from trl import evaluation


logger = logging.getLogger(__name__)


def make_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nx = len(x)
    ny = len(y)

    # ensure 2D
    x = x.reshape(nx, -1)
    y = y.reshape(ny, -1)

    x = np.tile(x, (1, ny)).reshape(-1, x.shape[1])
    y = np.tile(y, (nx, 1))
    return np.concatenate((x, y), axis=-1)


def rec_to_array(recarray: np.rec.array) -> np.ndarray:
    nrows = len(recarray)
    d = recarray.view(theano.config.floatX, np.ndarray)
    return d.reshape((nrows, len(d) // nrows))


def discretize_space(space: gym.Space, max=3):
    floatX = theano.config.floatX
    if isinstance(space, spaces.Discrete):
        return np.arange(space.n, dtype=floatX)

    if isinstance(space, spaces.Box):
        if space.shape[0] > 1:
            discretized_space = np.zeros((space.shape[0], max))
            for i in range(space.shape[0]):
                discretized_space[i, :] = np.linspace(space.low[i],
                                                      space.high[i],
                                                      max,
                                                      dtype=floatX)

            return cartesian(discretized_space.tolist())
        else:
            return np.linspace(space.low, space.high, max, dtype=floatX)

    # ifqi's DiscreteValued
    try:
        return space.values
    except AttributeError:
        pass

    raise NotImplementedError


def get_space_dim(space: gym.Space):
    return np.prod(getattr(space, 'shape', 1))


def load_dataset(filepath, name='dataset'):
    logger.info('Loading %s from %s', name, filepath)
    with closing(h5py.File(filepath, 'r')) as file:
        dataset = file[name][:]
        rec = file[name].attrs.get('rec', False)
        return np.rec.array(dataset, copy=False) if rec else dataset


def save_dataset(dataset, filepath, name='dataset'):
    with closing(h5py.File(filepath)) as file:
        if name in file:
            del file[name]
        file.create_dataset(name, data=dataset)
        file[name].attrs['rec'] = isinstance(dataset, np.recarray)
        file.flush()
    logger.info('Saved %s to %s', name, filepath)


def norm(x, p=2):
    "Norm function accepting both ndarray or tensor as input"
    if p == np.inf:
        return (x ** 2).max()
    x = x if p % 2 == 0 else abs(x)
    return  (x ** p).sum() ** (1. / p)
