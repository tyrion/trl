from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import logging

import numpy as np


logger = logging.getLogger(__name__)


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

    @contextmanager
    def save_params(self, params=None):
        oldpar = self.params
        if params is not None:
            self.params = params
        yield
        self.params = oldpar

    # compatibility with ifqi_pbo
    get_weights = lambda self: self.params
    set_weights = lambda self, w: setattr(self, 'params', w)
    count_params = lambda self: len(self.params)


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

        fit_kwargs = self.fit_kwargs
        if kwargs:
            fit_kwargs = fit_kwargs.copy()
            fit_kwargs.update(kwargs)

        #i = x.reshape(-1, self.input_dim)
        history = self.model.fit(x, y, **fit_kwargs)
        loss = history.history['loss'][-1]
        logger.info('Epochs: %d, loss: %f', len(history.epoch), loss)
        return history

    def predict(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model.predict(x)
