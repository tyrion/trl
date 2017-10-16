#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import pickle

import base
import run_many

from hyperopt import hp
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import callbacks, regularizers

import hyperopt
import keras
import numpy as np

from trl.algorithms import FQI
from trl import regressor, utils


logger = logging.getLogger('trl.algorithms')

class NeuralFQI(FQI):
    def __init__(self, *args, reinit=False, weight_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_steps = int(weight_steps)
        logger.info('NeuralFQI wit weight_steps=%d', self.weight_steps)
        # r = self.dataset.reward

        # c0 = np.count_nonzero(r == 0)
        # c1 = np.count_nonzero(r == 1)
        # c2 = np.count_nonzero(r == -1)

        # w = np.ones_like(r)
        # w[r == +1] = c0 / c1
        # w[r == -1] = c0 / c2
        # self.w = w

    def get_weights(self, y):
        _, idx, counts = np.unique(y, return_inverse=True, return_counts=True)
        w = (len(y) / (len(counts) * counts))[idx]
        return w

    def first_step(self, budget=None):
        y = self.dataset.reward
        w = self.get_weights(y) if self.weight_steps > 0 else None
        self.q.fit(self.SA, y, sample_weight=w)

    def step(self, i=0, budget=None):
        y = self.dataset.reward + self.gamma * self.max_q()
        w = self.get_weights(y) if self.weight_steps > i else None
        self.q.fit(self.SA, y, sample_weight=w)
        #log(i, y, self.params)



def get_params():
    #experiments = list(range(1, 21))
    experiments = [50]
    for n in experiments:
        yield 'sw12', {
            'training_iterations': n,
            'q_load_path': 'nn4',
            'evaluation_episodes': 289,
        }, {}


ACTIVATIONS = ['sigmoid', 'tanh', 'relu']
INITIALIZATIONS = [
    ('uniform', {'scale': 0.5}),
    'uniform', 'glorot_uniform', 'zero', 'one',
]
OPTIMIZERS = ['adadelta','adam', 'rmsprop']

space = {
    'training_iterations': 50,
    'evaluation_episodes': 289,
    'use_action_regressor': True,
    'env_name': 'CarOnHill-v0',
    'repeat': 20,

    'algorithm_config': {
        'reinit': False,
        'weight_steps': hp.qloguniform('weight_steps', 0, np.log(51), 1) -1,
    },

    'nb_epoch': 1000,
    'neurons': hp.qloguniform('neurons', np.log(4), np.log(64), 4),
    'batch_size': hp.qloguniform('batch_size', np.log(50), np.log(1000), 50),
    'activation': hp.choice('activation', ACTIVATIONS),
    'activity_regularizer': hp.choice('activity_regularizer', [True, False]),
    'init': hp.choice('init', INITIALIZATIONS),
    'loss': 'mse',
    'optimizer': hp.choice('optimizer', OPTIMIZERS),


    'early_stopping': {
        'monitor': 'loss',
        'min_delta': 1e-4,
        'patience': 20,
        'mode': 'auto',
    },
}


class HyperOptExperiment(run_many.Experiment):
    algorithm_class = NeuralFQI


    def _get_init(self, value):
        if isinstance(value, str):
            return value

        module = keras.initializations
        init_name, kw = value
        def init(shape, name=None):
            return getattr(module, init_name)(shape, name=name, **kw)

        return init


    def _get_layer_cfg(self, cfg):
        return {
            'output_dim': cfg['neurons'],
            'init': self._get_init(cfg['init']),
            'activation': cfg['activation'],
        }

    def get_q(self):
        cfg = self.config
        n = cfg['neurons']

        d1 = Dense(input_dim=self.input_dim, **self._get_layer_cfg(cfg))
        d2 = Dense(**self._get_layer_cfg(cfg))

        out_layer = self._get_layer_cfg(cfg)
        out_layer.update({'output_dim': 1, 'activation': 'linear'})
        if cfg['activity_regularizer']:
            out_layer['activity_regularizer'] = regularizers.l1(0.01)
        d3 = Dense(**out_layer)


        model = Sequential()
        model.add(d1)
        model.add(d2)
        model.add(d3)
        model.compile(loss=cfg['loss'], optimizer=cfg['optimizer'])

        cb = callbacks.EarlyStopping(verbose=0, **cfg['early_stopping'])

        return regressor.KerasRegressor(model, self.input_dim, callbacks=[cb], verbose=0,
                                        nb_epoch=cfg['nb_epoch'],
                                        batch_size=cfg['batch_size'])


def run(config):
    for k in ('neurons', 'batch_size'):
        config[k] = int(config[k])

    keys = ('neurons', 'batch_size', 'activation', 'activity_regularizer',
            'init', 'optimizer')
    for k in keys:
        print("%s=%s" % (k, config[k]), end=' ')
    print('weight_steps=%s' % config['algorithm_config']['weight_steps'])

    bytes = json.dumps(config, sort_keys=True).encode('ascii')
    name = hashlib.sha1(bytes).hexdigest()
    print(config)

    try:
        r = run_many.run_experiment(name, config,
            repeat=config['repeat'], workers=20, resume=True,
            seeds=seeds, experiment_class=HyperOptExperiment)
        score = r[:, 0]

        return {
            'loss': -score.mean(),
            'loss_variance': score.var(),
            'status': hyperopt.STATUS_OK,
            'hash': name,
        }
    except Exception:
        return {'status': hyperopt.STATUS_FAIL, 'hash': name}





def get_model(input_dim=3, output_dim=1, activation='sigmoid'):
    d1 = Dense(20, input_dim=input_dim, init='uniform', activation=activation)
    d2 = Dense(20, init='uniform', activation=activation)
    d3 = Dense(output_dim, init='uniform', activation='linear',
               activity_regularizer=regularizers.l1(0.01))

    model = Sequential()
    model.add(d1)
    model.add(d2)
    model.add(d3)
    model.compile(loss='mse', optimizer='adam')

    return model


def get_q(input_dim=3, output_dim=1):
    model = get_model(input_dim, output_dim, activation)

    cb = callbacks.EarlyStopping(monitor='loss', min_delta=1e-4,
                                 patience=20, mode='auto', verbose=0)

    return regressor.KerasRegressor(model, input_dim, callbacks=[cb], verbose=0,
                                    nb_epoch=config['nb_epoch'],
                                    batch_size=config['batch_size'])


class Experiment(run_many.Experiment):
    env_name = 'CarOnHill-v0'
    algorithm_class = NeuralFQI
    training_iterations = 50
    evaluation_episodes = 289
    use_action_regressor = True

    def get_q(self):
        return get_q(self.input_dim)


def main():
    trials_fp = 'trials.pickle'
    try:
        with open(trials_fp, 'rb') as fp:
            trials = pickle.load(fp)
        print(len(trials))
    except:
        trials = hyperopt.Trials()


    try:
        hyperopt.fmin(run, space=space, algo=hyperopt.tpe.suggest,
                      max_evals=100, verbose=2, trials=trials, return_argmin=0)
    except SystemExit:
        print('Interrupt')


    if len(trials) < 1:
        return

    tt = trials._dynamic_trials[-1]
    if tt['result']['status'] not in (hyperopt.STATUS_OK, hyperopt.STATUS_FAIL):
        print('discarding last trial')
        trials._ids.discard(tt['tid'])
        del trials._dynamic_trials[-1]
        trials.refresh()

    if len(trials) < 1: # should count the successful statuses
        return

    print(trials.argmin)
    print(trials.best_trial['result'])

    with open(trials_fp, 'wb') as fp:
        pickle.dump(trials, fp)


def nope():

    if N > 1:
        run_many.run_many(repeat=N, params=get_params, workers=20, resume=True,
                          seeds=seeds, experiment_class=Experiment)
    else:
        Experiment(dataset_load_path='dataset-9.h5',
                   save_regressor_path='single.h5',
                   save_trace_path='single.h5').run()

if __name__ == '__main__':

    import signal
    import sys
    import logging

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1


    def handler(signum, frame):
        #logging.critical('Received Interrupt. Terminating')
        raise SystemExit

    signal.signal(signal.SIGINT, handler)

    logging.config.dictConfig(base.LOGGING)


    seeds = utils.load_dataset('seeds.h5')

    from hyperopt.pyll import stochastic
    config = {
        'use_action_regressor': True,
        'neurons': 12.0,
        'early_stopping': {
            'min_delta': 0.0001,
            'patience': 20,
            'monitor': 'loss',
            'mode': 'auto'
        },
        'nb_epoch': 5,
        'weights': 4.0,
        'reset_weights': False,
        'activity_regularizer': False,
        'evaluation_episodes': 289,
        'init': 'zero',
        'optimizer': 'adadelta',
        'env_name': 'CarOnHill-v0',
        'repeat': 2,
        'loss': 'mse',
        'training_iterations': 2,
        'activation': 'sigmoid',
        'batch_size': 300.0
    }
    #config = stochastic.sample(space)

    main()
