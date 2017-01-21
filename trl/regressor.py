from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import numpy as np


class Regressor(metaclass=ABCMeta):

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

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

    def __init__(self, model, input_dim=2):
        self.model = model
        self.input_dim = input_dim
        self._params = self._shape = None

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

    def fit(self, x, y):
        self._params = None
        #i = x.reshape(-1, self.input_dim)
        return self.model.fit(x, y, nb_epoch=5, verbose=2)

    def predict(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model.predict(x)
