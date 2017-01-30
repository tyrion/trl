from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import base64
import copy
import io
import logging
import pickle

import numpy as np
import h5py
from sklearn.externals import joblib
from sklearn import preprocessing
from ifqi.models import actionregressor


logger = logging.getLogger(__name__)


# TODO unifiy this with utils.load_dataset?
def load_regressor(filepath, name='regressor'):
    f = h5py.File(filepath, 'r')
    regressor = f[name]
    cls = _loads(regressor.attrs['class'])
    try:
        return cls.load(regressor)
    finally:
        logger.info('Loaded regressor from %s', filepath)
        f.close()


def save_regressor(regressor, filepath, name='regressor', attrs=None):
    f = h5py.File(filepath)
    try:
        data = regressor.save(f)
        if name in f:
            del f[name]
        f[name] = data
        if attrs is not None:
            for k, v in attrs.items():
                f[name].attrs[k] = v
        f.flush()
    finally:
        f.close()

    logger.info('Saved regressor to %s', filepath)


def _dumps(object):
    return base64.b64encode(pickle.dumps(object))


def _loads(object):
    return pickle.loads(base64.b64decode(object))


# XXX dirty hack, monkey patching h5py
@contextmanager
def _patch_h5(group):
    group.flush = lambda: None
    group.close = lambda: None
    File = h5py.File
    h5py.File = lambda *a, **kw: group
    yield
    h5py.File = File
    del group.flush
    del group.close


class Regressor(metaclass=ABCMeta):
    scaler_x = None
    scaler_y = None

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def predict_one(self, x):
        return self.predict(x[np.newaxis, :])[0]

    def __call__(self, x):
        return self.predict(x)

    def transform(self, x, y):
        self.scaler_x = s_x = preprocessing.StandardScaler(copy=True)
        self.scaler_y = s_y = preprocessing.StandardScaler(copy=True)

        # ensure 2d
        y = y.reshape(len(y), -1)

        return s_x.fit_transform(x), s_y.fit_transform(y)

    def save(self, group):
        regressor = group.create_dataset(None, data=self.params)
        regressor.attrs['class'] = _dumps(self.__class__)
        return regressor

    @classmethod
    def load(cls, dataset):
        return cls(dataset[:])

    @contextmanager
    def save_params(self, params=None):
        oldpar = self.params
        if params is not None:
            self.params = params
        yield
        self.params = oldpar

    # compatibility with ifqi
    get_weights = lambda self: self.params
    set_weights = lambda self, w: setattr(self, 'params', w)
    count_params = lambda self: len(self.params)


class KerasRegressor(Regressor):

    def __init__(self, model, input_dim=2, **fit_kwargs):
        self._model = model
        self.input_dim = input_dim
        self._params = self._shape = None
        self.fit_kwargs = fit_kwargs
        fit_kwargs.setdefault('verbose', 0)

        self.scaler_x = None
        self.scaler_y = None

    @property
    def params(self):
        if self._params is None:
            weights = self._model.get_weights()
            self._shapes = [w.shape for w in weights]

            self._params = np.concatenate([w.ravel() for w in weights])
            self._params.flags['WRITEABLE'] = False
        return self._params

    @params.setter
    def params(self, weights):
        weights = weights.ravel()
        if self._shapes is None:
            self.params

        result = []
        i = 0
        for shape in self._shapes:
            n = np.prod(shape)
            w = weights[i:i+n].reshape(shape)
            result.append(w)
            i += n

        self._model.set_weights(result)
        self._params = weights

    @property
    def trainable_weights(self):
        return self._model.trainable_weights

    @property
    def inputs(self):
        return self._model.inputs

    @property
    def outputs(self):
        return self._model.outputs


    def fit(self, x, y, **kwargs):
        self._params = None
        x, y = self.transform(x, y)

        if self.fit_kwargs:
            fit_kwargs = copy.deepcopy(self.fit_kwargs)
            fit_kwargs.update(kwargs)
        else:
            fit_kwargs = kwargs

        #i = x.reshape(-1, self.input_dim)
        history = self._model.fit(x, y, **fit_kwargs)
        loss = history.history['loss'][-1]
        logger.info('Epochs: %d, loss: %f', len(history.epoch), loss)
        return history

    def predict(self, x):
        x = x.reshape(-1, self.input_dim)
        x = x if self.scaler_x is None else self.scaler_x.transform(x)

        y = self._model.predict(x)
        y = y if self.scaler_y is None else self.scaler_y.inverse_transform(y)

        return y

    def model(self, inputs, params=None):
        out = inputs
        for el in self._model.flattened_layers:
            out = el(out)
        return out

    def get_config(self):
        config = copy.deepcopy(self.fit_kwargs)
        config['input_dim'] = self.input_dim
        return config

    def save(self, group):
        group = group.create_group(None)
        group.attrs['class'] = _dumps(self.__class__)
        group.attrs['config'] = _dumps(self.get_config())

        with _patch_h5(group):
            self._model.save('whatever', overwrite=True)

        return group

    @classmethod
    def load(cls, group):
        from keras.models import load_model
        config = _loads(group.attrs['config'])

        with _patch_h5(group):
            model = load_model('whatever')

        return cls.from_config(model, config)

    @classmethod
    def from_config(cls, model, config):
        return cls(model, **config)


class SkLearnRegressorMixin(Regressor):

    @property
    def params(self):
        return self.get_params()

    @params.setter
    def params(self, params):
        return self.set_params(**params)

    def save(self, group):
        bytes = io.BytesIO()
        joblib.dump(self, bytes)
        data = np.frombuffer(bytes.getbuffer(), np.uint8)

        regressor = group.create_dataset(None, data=data)
        regressor.attrs['class'] = _dumps(self.__class__)
        return regressor

    @classmethod
    def load(cls, dataset):
        return joblib.load(io.BytesIO(dataset[:]))


class ActionRegressor(Regressor):

    def __init__(self, regressors, actions):
        if isinstance(regressors, Regressor):
            regressors = [copy.deepcopy(regressors) for _ in actions]

        self.regressors = regressors
        self.actions = actions.reshape(len(actions), -1)
        self.action_dim = self.actions.shape[-1]

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, params):
        raise NotImplementedError

    def _regressors(self, actions):
        for action, regressor in zip(self.actions, self.regressors):
            i = np.all(actions == action, axis=1)
            if np.any(i):
                yield regressor, i

    def fit(self, x, y):
        states, actions = np.hsplit(x, [-self.action_dim])
        for regressor, i in self._regressors(actions):
            regressor.fit(states[i], y[i])

    def predict(self, x):
        states, actions = np.hsplit(x, [-self.action_dim])
        predictions = np.zeros(len(x))
        for regressor, i in self._regressors(actions):
            predictions[i] = regressor.predict(states[i])
        return predictions

    def save(self, group):
        group = group.create_group(None)
        for i, regressor in enumerate(self.regressors):
            group['regressor-{}'.format(i)] = regressor.save(group)

        group.create_dataset('actions', data=self.actions)
        group.attrs['class'] = _dumps(self.__class__)
        return group

    @classmethod
    def load(cls, group):
        regressors = [_loads(v.attrs['class']).load(v) for (k,v)
                      in sorted(group.items()) if k.startswith('reg')]
        actions = group['actions'][:]
        return cls(regressors, actions)
