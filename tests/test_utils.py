import numpy as np
import pytest
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



def test_rec_to_array_simple():
    a = np.zeros(5)
    b = utils.rec_to_array(a)

    # works but makes the array 2d
    assert b.shape == (5, 1)

    a = np.zeros((5,5,5))
    # you should really only pass rec arrays
    with pytest.raises(ValueError):
        b = utils.rec_to_array(a)



def test_rec_to_array_structured():
    a = np.zeros(2, 'f4,2f4')
    b = utils.rec_to_array(a)

    assert b.dtype == np.dtype('f4')
    assert b.shape == (2, 3)

    a = np.zeros(2, dtype='2f4,f4')
    b = utils.rec_to_array(a)

    assert b.dtype == np.dtype('f4')
    assert b.shape == (2, 3)



def test_rec_to_array_record():
    a = np.recarray((2,), 'f4,2f4')
    b = utils.rec_to_array(a)

    assert b.dtype == np.dtype('f4')
    assert b.shape == (2, 3)

    a = np.recarray((2,), dtype='2f4,f4')
    b = utils.rec_to_array(a)

    assert b.dtype == np.dtype('f4')
    assert b.shape == (2, 3)


def test_rec_to_array_dtype():
    """test that dtype is taken from first element"""
    a = np.array([(0, [1, 2]), (1, [3, 2**50])], dtype='i4,2i8')
    b = utils.rec_to_array(a)

    assert b.dtype == np.dtype('i4')

    # shape is (2, 5) because the 8 bytes ints (i8) are being viewed as i4
    assert b.shape == (2, 5)