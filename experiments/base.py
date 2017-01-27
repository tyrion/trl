import os
import logging
import logging.config

os.environ.setdefault('KERAS_BACKEND', 'theano')

import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import ExtraTreesRegressor

from trl import algorithms, ifqi, regressor
from trl.experiment import Experiment


LOGGING = {
    'version': 1,
    'formatters': {
	'default': {
	    'format': '%(asctime)s %(levelname)5s:%(name)s: %(message)s',
	},
    },
    'handlers': {
	'console': {
	    'class': 'logging.StreamHandler',
	    'level': 'INFO',
	    'formatter': 'default',
	},
    },
    'loggers': {
	'trl': {
	    'level': 'DEBUG',
	},
    },
    'root': {
	'level': 'DEBUG', # debug
	'handlers': ['console'],
    },
}


class CurveFitQRegressor(regressor.Regressor):

    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params-0.0001)

    def Q(self, sa, b, k):
        s, a = sa[:, 0], sa[:, 1]
        return - b * b * s * a - 0.5 * k * a * a - 0.4* k * s * s

    def predict(self, x):
        return self.Q(x, *self.params)


class ExtraTreesRegressor(ExtraTreesRegressor, regressor.SkLearnRegressorMixin):
    pass


def build_nn(input_dim=2, output_dim=2, activation='sigmoid'):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import callbacks

    model = Sequential()
    model.add(Dense(2, input_dim=input_dim, init='uniform',
                    activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return regressor.KerasRegressor(model, input_dim)


def build_nn2(input_dim=2, output_dim=2, activation='sigmoid'):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import callbacks

    cb = callbacks.EarlyStopping(monitor='loss', min_delta=6e-1,
                                 patience=5, mode='auto')

    model = Sequential()
    model.add(Dense(4, input_dim=input_dim, init='uniform', activation=activation))
    model.add(Dense(4, init='uniform', activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return regressor.KerasRegressor(model, input_dim, callbacks=[cb],
                                    nb_epoch=50, batch_size=100)


def build_curve_fit(input_dim=2, output_dim=1):
    return CurveFitQRegressor(np.array([0.0,0.0]))


def build_extra_trees(input_dim=2, output_dim=1):
    return ExtraTreesRegressor(n_estimators=50, min_samples_split=5,
                               min_samples_leaf=2, criterion='mse')


def handler(signum, frame):
    logging.critical('Received Interrupt. Terminating')
    raise SystemExit


class CLIExperiment(Experiment):
    q_load_path = 'curve_fit'

    def get_q(self):
        q = self.q_load_path
        build = globals().get('build_{}'.format(q), None)
        if build is None:
            return super().get_q()
        return build(input_dim=self.input_dim, output_dim=1)

    def get_algorithm_config(self):
        if self.algorithm_class in (algorithms.NESPBO, ifqi.PBO):
            dim = len(self.q.params)
            self.algorithm_config['bo'] = build_nn(input_dim=dim, output_dim=dim)
        return self.algorithm_config

    @classmethod
    def _setup(cls, i, logfile='experiment-{i}.log', **kwargs):
        LOGGING['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'default',
            'filename': logfile.format(i=i),
            'mode': 'w',
        }
        LOGGING['root']['handlers'] = ['file']
        logging.config.dictConfig(LOGGING)
