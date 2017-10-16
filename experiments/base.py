import os
import logging
import logging.config
import math

os.environ.setdefault('KERAS_BACKEND', 'theano')

import theano

floatX = theano.config.floatX
import keras # this will override theano.config.floatX

# respect theano settings.
keras.backend.set_floatx(floatX)
theano.config.floatX = floatX

import numpy as np
from theano import tensor as T
from scipy.optimize import curve_fit
from sklearn.ensemble import ExtraTreesRegressor

from trl import algorithms, regressor, utils
from trl.algorithms import ifqi
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
        'level': 'DEBUG',
        'handlers': ['console'],
    },
}

ALGORITHMS = {
    'fqi': algorithms.FQI,
    'wfqi': algorithms.WeightedFQI,
    'pbo': algorithms.NESPBO,
    'gradfqi': algorithms.GenGradFQI,
    'gradpbo': algorithms.GradPBO,
    'ifqi_fqi': ifqi.FQI,
    'ifqi_pbo': ifqi.PBO,
    'ifqi_gradpbo': ifqi.GradPBO,
}


class CurveFitQRegressor(regressor.TheanoRegressor):
    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params - 0.0001)

    def Q(self, sa, b, k):
        return self._model(sa, [b, k])

    def _model(self, sa, theta):
        s = sa[:, 0]
        a = sa[:, 1]
        b = theta[0]
        k = theta[1]
        return - b * b * s * a - 0.5 * k * a * a - 0.4* k * s * s


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


def build_nn2(input_dim=2, output_dim=2, activation='tanh'):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import callbacks

    cb = callbacks.EarlyStopping(monitor='loss', min_delta=1e-3,
                                 patience=5, mode='auto')

    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, init='glorot_uniform', activation=activation))
    model.add(Dense(8, init='glorot_uniform', activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return regressor.KerasRegressor(model, input_dim, callbacks=[cb],
                                    nb_epoch=100, batch_size=100)


def build_nn3(input_dim=2, output_dim=2, activation='sigmoid'):
    """
    Good for CartPole-v0
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import callbacks

    cb = callbacks.EarlyStopping(monitor='loss', min_delta=0.58,
                                 patience=600, mode='auto')

    model = Sequential()
    model.add(Dense(4, input_dim=input_dim, init='uniform', activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return regressor.KerasRegressor(model, input_dim, callbacks=[cb],
                                    nb_epoch=100, batch_size=2000)


def build_nn4(input_dim=2, output_dim=2, activation='tanh'):
    """
    Good for CarOnHill-v0
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import callbacks, regularizers

    cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
                                 patience=20, mode='auto', verbose=0)

    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, init='uniform',
                    activation=activation,
                    #W_regularizer=regularizers.l2(0.01),
                    #activity_regularizer=regularizers.l1(0.01)
    ))
    #model.add(Dense(10, init='uniform', activation=activation,
                    #W_regularizer=regularizers.l2(0.01),
                    #activity_regularizer=regularizers.l1(0.01)
    #))
    #model.add(Dropout(0.1))
    model.add(Dense(output_dim, init='uniform', activation='linear'))#, activity_regularizer=regularizers.l1(0.01)))
    model.compile(loss='mse', optimizer='rmsprop')
    return regressor.KerasRegressor(model, input_dim, callbacks=[cb],
                                    validation_split=0.3, verbose=0,
                                    nb_epoch=300, batch_size=100)


def _uniform05(shape, name):
    return keras.initializations.uniform(shape, scale=0.5, name=name)


def build_nfqi(input_dim=2, output_dim=2):
    """
    NN taken from NFQI paper.
    """
    #print('BUILD_NFQI', input_dim, output_dim)
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import callbacks

    activation = 'sigmoid'
    init = _uniform05
    cb = callbacks.EarlyStopping(monitor='loss', min_delta=0.03,
                                 patience=20, mode='auto')

    model = Sequential()
    model.add(Dense(5, input_dim=input_dim, init=init, activation=activation))
    model.add(Dense(5, init=init, activation=activation))
    model.add(Dense(output_dim, init=init, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')

    return regressor.KerasRegressor(model, input_dim, callbacks=[cb],
                                    nb_epoch=500, batch_size=32)

def build_bo(q):
    from keras.models import Model
    from keras.layers.core import Dense
    from trl import utils
    inputs = utils.t_make_inputs(q.trainable_weights, dtype=theano.config.floatX)
    last_layer_dim = sum([v.get_value().size for v in q.trainable_weights])
    #print(last_layer_dim)
    #print(inputs)
    #print([el.ndim for el in inputs])
    c = utils.k_concat(inputs)
    d1 = Dense(5, init=_uniform05, activation='sigmoid', name='d1')(c)
    d2 = Dense(5, init=_uniform05, activation='sigmoid', name='d2')(d1)
    d3 = Dense(last_layer_dim, init=_uniform05, activation='linear', name='d3')(
        d2)
    o = utils.Split(inputs)(d3)
    bo = Model(input=inputs, output=o)
    return regressor.KerasRegressor(bo, None)


def build_curve_fit(input_dim=2, output_dim=1):
    return CurveFitQRegressor(np.array([0.0,0.0], dtype=theano.config.floatX))


def build_extra_trees(input_dim=2, output_dim=1):
    return ExtraTreesRegressor(n_estimators=50, min_samples_split=5,
                               min_samples_leaf=2, criterion='mse')


def handler(signum, frame):
    logging.critical('Received Interrupt. Terminating')
    raise SystemExit


def bo(q):
    dim = len(q.params)
    return build_nn(input_dim=dim, output_dim=dim)


class CLIExperiment(Experiment):
    q_load_path = 'curve_fit'
    budget = 1

    def get_q(self):
        q = self.q_load_path
        build = globals().get('build_{}'.format(q), None)
        if build is None:
            return super().get_q()
        return build(input_dim=self.input_dim, output_dim=1)

    def get_algorithm_config(self):
        config = super().get_algorithm_config()
        if self.algorithm_class in (algorithms.NESPBO, ifqi.PBO,
                                    algorithms.GradPBO, ifqi.GradPBO):
            dim = len(self.q.params)
            #config['bo'] = build_nn(input_dim=dim, output_dim=dim)
            config['bo'] = build_bo(self.q)
        # if self.algorithm_class == algorithms.GradPBO:
        #     config.update({
        #         'update_loss': 'be',
        #         'update_steps': 200,
        #         'batch_size': 100,
        #     })
        return config

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

    def get_training_episodes(self):
        if self.env_name != 'CarPole-v0':
            return super().get_training_episodes()
        n = self.config.get('training_episodes', 2000)
        unw = self.env.unwrapped
        x = unw.x_threshold
        t = 0.95 * unw.theta_threshold_radians
        return np.random.uniform([-x, -3.5, -t, -3], [x, 3.5, t, 3], (n, 4))

    def get_evaluation_episodes(self):
        if self.env_name != 'CarOnHill-v0':
            return super().get_evaluation_episodes()

        n = self.config.get('evaluation_episodes', self.__class__.evaluation_episodes)
        if n <= 0:
            return n
        n = n ** 0.5
        return utils.make_grid(np.linspace(-1, 1, n), np.linspace(-3, 3, n))
