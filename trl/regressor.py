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
from ifqi.models import actionregressor


logger = logging.getLogger(__name__)


def load_regressor(filepath, name='regressor'):
    f = h5py.File(filepath, 'r')
    regressor = f[name]
    cls = _loads(regressor.attrs['class'])
    try:
        return cls.load(regressor)
    finally:
        logger.info('Loaded regressor from %s', filepath)
        f.close()


def save_regressor(regressor, filepath, name='regressor'):
    f = h5py.File(filepath)
    try:
        data = regressor.save(f)
        if name in f:
            del f[name]
        f[name] = data
        f.flush()
    finally:
        f.close()

    logger.info('Saved regressor to %s', filepath)


def _dumps(object):
    return base64.b64encode(pickle.dumps(object))

def _loads(object):
    return pickle.loads(base64.b64decode(object))


class Regressor(metaclass=ABCMeta):

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


class KerasRegressor(Regressor):

    def __init__(self, model, input_dim=2, **fit_kwargs):
        self.model = model
        self.input_dim = input_dim
        self._params = self._shape = None
        self.fit_kwargs = fit_kwargs
        fit_kwargs.setdefault('verbose', 0)

    @property
    def params(self):
        if self._params is None:
            weights = self.model.get_weights()
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

        self.model.set_weights(result)
        self._params = weights

    def fit(self, x, y, **kwargs):
        self._params = None

        if self.fit_kwargs:
            fit_kwargs = copy.deepcopy(self.fit_kwargs)
            fit_kwargs.update(kwargs)
        else:
            fit_kwargs = kwargs

        #i = x.reshape(-1, self.input_dim)
        history = self.model.fit(x, y, **fit_kwargs)
        loss = history.history['loss'][-1]
        logger.info('Epochs: %d, loss: %f', len(history.epoch), loss)
        return history

    def predict(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model.predict(x)

    def get_config(self):
        config = copy.deepcopy(self.fit_kwargs)
        config['input_dim'] = self.input_dim
        return config

    def save(self, group):
        group = group.create_group(None)
        group.attrs['class'] = _dumps(self.__class__)
        group.attrs['config'] = _dumps(self.get_config())

        with _patch_h5(group):
            self.model.save('whatever', overwrite=True)

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
