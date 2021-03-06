import collections
import functools
import logging
from contextlib import closing

import h5py
import gym
import keras
import numpy as np
import theano
from gym import spaces
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Reshape, merge
from keras.models import Model

from trl import evaluation


logger = logging.getLogger(__name__)
floatX = theano.config.floatX


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
    dtype = recarray.dtype
    while dtype.fields is not None:
        dtype = dtype[0]
    dtype = dtype.base

    nrows = len(recarray)
    d = recarray.view(dtype, np.ndarray)
    return d.reshape((nrows, len(d) // nrows))


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=floatX):
    """
    Linspace with support for multidimensional arrays.
    """
    start = np.asanyarray(start, dtype)
    stop = np.asanyarray(stop, dtype)
    assert start.shape == stop.shape

    div = np.asanyarray((num - 1) if endpoint else num, dtype)

    shape = (num,) + start.shape
    step = (stop - start) / div

    data = np.tile(step.ravel(), num).reshape(shape)
    data = (data.T * np.arange(0, num, dtype=dtype)).T + start
    return (data, step) if retstep else data


def discretize_space(space: gym.Space, max=20):
    if isinstance(space, spaces.Discrete):
        return np.arange(space.n)

    if isinstance(space, spaces.Box):
        d = linspace(space.low, space.high, max, dtype=floatX)
        return d.ravel() if space.shape == (1,) else d

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
        dataset = file[name].value
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



def t_make_inputs(inputs, dtype=theano.config.floatX):
    # ndim(input) = ndim(v) + 1 to account for batch size
    return [Input(shape=v.get_value().shape, name=v.name, dtype=dtype)
            for v in inputs]


def k_concat(inputs):
    if len(inputs) == 1:
        return inputs
    fn = lambda v, s: v if len(s) <3 else Reshape((np.prod(s[1:]),))(v)
    return merge([fn(v, v._keras_shape) for v in inputs], mode='concat')


def k_init(init):
    """
    Return a keras initialization function.

    >>> k_init('glorot_uniform')
    >>> k_init('uniform', scale=0.5)
    """
    (name, kwargs) = (init, {}) if isinstance(init, str) else init
    fn = getattr(keras.initializations, name)
    if kwargs:
        fn = functools.update_wrapper(functools.partial(fn, **kwargs), fn)
    return fn


def k_layers(inputs, *, layers, neurons, activation, init, output_dim=1):
    """
    Build Dense layers to be used in a Keras model.
    """
    assert layers >= 1, "Layers must be greater than 1"
    n = neurons // layers
    x = inputs
    for layer in range(layers):
        x = Dense(n, activation=activation, init=k_init(init))(x)
    return Dense(output_dim, activation='linear', init=k_init(init))(x)


def build_bo(*, layers, neurons, activation, init):
    from trl import regressor

    def bo(q):
        inputs = t_make_inputs(q.trainable_weights)
        output_dim = sum([v.get_value().size for v in q.trainable_weights])
        c = k_concat(inputs)

        outputs = k_layers(
            c, layers=layers, neurons=neurons, activation=activation,
            init=init, output_dim=output_dim)
        outputs = Split(inputs)(outputs)
        model = Model(input=inputs, output=outputs)
        return regressor.KerasRegressor(model, None)

    return bo


class Split(Layer):
    """
    >>> inputs = t_make_inputs(q.trainable_weights)
    >>> c = k_concat(inputs)

    >>> d1 = Dense(10, init='uniform', activation='sigmoid', name='d1')(c)
    >>> d2 = Dense(10, init='uniform', activation='linear', name='d2')(d1)

    >>> o = Split(inputs)(d2)

    >>> model = Model(input=inputs, output=o)
    """
    def __init__(self, original, **kw):
        self.original = original
        super().__init__(**kw)

    def _call(self, x):
        i = j = 0
        for v in self.original:
            shape = v._keras_shape[1:]
            j += np.prod(shape)
            v = x[:, i:j]
            yield v if len(shape) < 2 else K.reshape(v, (-1,) + shape)
            #yield v
            i = j

    def call(self, x, mask=None):
        return list(self._call(x))

    def get_output_shape_for(self, input_shape):
        return [v.shape for v in self.original]

    def compute_mask(self, input, input_mask=None):
        return [None] * len(self.original)


def rec_update(d, u):
    """
    Recursive dict update by Alex Martelli.
    See https://stackoverflow.com/a/3233356/641317
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = rec_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d