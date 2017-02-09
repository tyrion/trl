import numpy as np
import theano
from gym import spaces

from trl import utils


def test_discretize_space_discrete():
    space = spaces.Discrete(10)
    assert np.all(utils.discretize_space(space, 1) == np.arange(10))


def test_discretize_space_box0d():
    space = spaces.Box(low=-10, high=10, shape=())
    ds = utils.discretize_space(space, 20)
    assert ds.shape == (20,)
    assert ds.dtype == theano.config.floatX


def test_discretize_space_box1d():
    space = spaces.Box(low=-10, high=10, shape=(1,))
    ds = utils.discretize_space(space, 20)
    assert ds.shape == (20,)
    assert ds.dtype == theano.config.floatX


def test_discretize_space_boxnd():
    shape = (5,4,7)
    space = spaces.Box(low=-10, high=10, shape=shape)
    ds = utils.discretize_space(space, 20)
    assert ds.shape == (20, ) + shape
    assert ds.dtype == theano.config.floatX

