#!/usr/bin/env python3

import hashlib
import logging
import time
import concurrent.futures
from contextlib import closing

import base
import run_many

import h5py
import keras.metrics
import numpy as np
import theano

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import callbacks, regularizers

from sklearn import metrics, preprocessing
from theano import tensor as T

from trl.algorithms import FQI
from trl import regressor, utils

base.LOGGING['formatters']['default']['datefmt'] = '%H:%M:%S'
logging.config.dictConfig(base.LOGGING)


def get_q(input_dim=3, output_dim=1, activation='tanh', metrics=()):
    activation = 'tanh'
    layers = [
        Dense(64, input_dim=input_dim, init='uniform', activation=activation),
        Dense(64, init='uniform', activation=activation),
        Dense(output_dim, init='uniform', activation='linear')
    ]
    # layers = [
    #     Dense(64, input_dim=input_dim, init='uniform', activation=activation),
    #     Dense(64, init='uniform', activation=activation),
    #     Dense(output_dim, init='uniform', activation='linear'),
    # ]

    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(loss='mse', optimizer='adam', metrics=metrics)

    return model


def handler(signum, frame):
    logging.critical('Received Interrupt. Terminating')
    raise SystemExit


def search_sorted(known, test_array):
    middles = known[1:] - np.diff(known)/2
    idx = np.searchsorted(middles, test_array)
    return idx


def values_at(i, gamma=0.95):
    z = np.zeros(i * 2 + 1)
    x = np.arange(i)
    z[:i] = -np.power(gamma, x)
    z[i+1:] = np.power(gamma, np.flipud(x))
    return z


def init_seed(seed, max_bytes=8):
    if seed is None:
        return int.from_bytes(os.urandom(max_bytes), 'big')
    return int(seed) % 2 ** (8 * max_bytes)


def get_seed(seed, stage, max_bytes=8):
    seed = (seed + stage) % 2 ** (8 * max_bytes)
    seed = seed.to_bytes(max_bytes, 'big')
    return hashlib.sha512(seed).digest()[-max_bytes:]


class Metrics(keras.callbacks.Callback):
    def __init__(self, x, y, y_values, scaler, w=None, verbose=1):
        super().__init__()
        self.x = x
        self.y = y
        self.y_values = y_values
        self.w = w
        self.scaler = scaler
        self.verbose = verbose
        self.history = []
        self.cls_true = search_sorted(self.y_values, self.y)
        self.classes = [str(x) for x in self.y_values]

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', .0) if logs else .0
        pred_s = self.model.predict(self.x)
        pred = self.scaler.inverse_transform(pred_s)

        self.cls_pred = search_sorted(self.y_values, pred)

        e1 = metrics.mean_squared_error(self.y, pred, sample_weight=self.w)
        e2 = metrics.accuracy_score(self.cls_true, self.cls_pred, sample_weight=self.w)

        # if len(self.y_values) > 5:
        #     e3 = 'not shown'
        # else:
        #     e3 = metrics.confusion_matrix(self.cls_true, cls_pred)
        self.history.append((loss, e1, e2))

        if self.verbose > 1:
            self.log()

    def log(self):
        h = self.history[-1]
        #cm = str(cm).replace('\n', ' ')
        e = len(self.history) -1
        logging.info('ep: %3d, loss: %f, mse: %f, acc: %f' % (e, *h))

    def on_train_end(self, logs=None):
        if not self.verbose:
            return

        labels = np.arange(len(self.y_values))
        report  = metrics.classification_report(self.cls_true, self.cls_pred,
                                                sample_weight=self.w,
                                                labels=labels, digits=6,
                                                target_names=self.classes)
        logging.info('\n'+report)
        #count_true = np.bincount(self.cls_true,

class Experiment:

    def __init__(self, i, seed=None, outfile='exp.h5', gamma=0.95):
        self.i = i
        self.outfile = outfile
        self.gamma = gamma
        self._seed = init_seed(seed)

    def setup(self):
        logging.info('Setup | seed: %s', self.seed(-1))
        self.actions = np.array([-4, 4])
        self.dataset = utils.load_dataset('dataset-%d.h5' % self.i)

        self.scaler_x = preprocessing.StandardScaler()
        self.scaler_y = preprocessing.StandardScaler()

        self._SA = utils.rec_to_array(self.dataset[['state', 'action']])
        self.SA = self.scaler_x.fit_transform(self._SA)

        self._S1A = utils.make_grid(self.dataset.next_state, self.actions)
        self.S1A = self.scaler_x.transform(self._S1A)

        self.q = get_q()

    def run(self, n, i=None):
        self.setup()
        # try action regressor
        if i is not None:
            seed = self.load_q('init', opts=['seed'])['seed']
            self._seed = init_seed(seed)
            self.init_weights = self.q.get_weights()

            self.load_q(i=i)
            logging.info('Loaded reg-%d' % i)
            max_q = self.max_q()
            i += 1
        else:
            i = 0
            max_q = 0
            self.init_weights = self.q.get_weights()
            self.save_q('init', seed=self._seed)
        y_values = values_at(i)

        for i in range(i, n):
            logging.info('Iteration %02d | seed: %s', i, self.seed(i))
            max_q = y_values[search_sorted(y_values, max_q)]
            assert i == 0 or np.any(max_q != 0)
            y = self.dataset.reward + self.gamma * max_q
            y_values = values_at(i+1)


            _, idx, counts = np.unique(y, return_inverse=True, return_counts=True)
            #w = (counts.max() / counts)[idx]
            w = (len(y) / (len(counts) * counts))[idx]
            m = Metrics(self.SA, y, y_values, self.scaler_y, w)

            #self.q.set_weights(self.init_weights)
            step = getattr(self, 'step%d' % i, self.step)
            step(y, sample_weight=w, callbacks=[m])

            self.save_q(i=i, scaler=regressor._dumps(self.scaler_y))

            # save also max_q
            max_q = self.max_q()
            self.save_dataset(max_q, 'max_q', i)

        # save final regressor as trl KerasRegressor
        r = regressor.KerasRegressor(self.q, 3, self.scaler_x, self.scaler_y)
        regressor.save_regressor(r, self.outfile, 'q')


    def max_q(self):
        n = len(self.dataset)
        n_actions = 2

        y = self.q.predict(self.S1A)
        y = self.scaler_y.inverse_transform(y)
        y = y.reshape((n, n_actions))

        y = y * (1 - self.dataset.absorbing)[:, np.newaxis]
        y = y.max(axis=1)

        return y

    def step(self, y, **fit_kwargs):
        y = y.reshape(-1, 1) # ensure 2d
        y = self.scaler_y.fit_transform(y)

        es = callbacks.EarlyStopping(monitor='loss', min_delta=1e-4,
                                     patience=20, mode='auto', verbose=0)

        cbs = [es]
        cbs.extend(fit_kwargs.get('callbacks', ()))
        fit_kwargs['callbacks'] = cbs

        fit_kwargs.setdefault('nb_epoch', 1000)
        fit_kwargs.setdefault('batch_size', 500)
        fit_kwargs.setdefault('verbose', 0)

        h = self.q.fit(self.SA, y, **fit_kwargs)
        loss = h.history['loss']
        logging.info('%03d Epochs, loss %f' % (len(loss), loss[-1]))
        return h, cbs

    def seed(self, stage):
        np_seed = get_seed(self._seed, stage)
        np_seed = [int.from_bytes(bytes(b), 'big')
                    for b in zip(*[iter(np_seed)]*4)]
        np.random.seed(np_seed)
        return np_seed

    ## LOAD/SAVE METHODS

    def save_q(self, path='reg', i=None, **opts):
        regressor.save_regressor(self, self.outfile, self._get_path(path, i),
                                 opts)

    def save_dataset(self, dataset, path, i=None):
        utils.save_dataset(dataset, self.outfile, self._get_path(path, i))

    def _get_path(self, path, i=None):
        path = 'it-%02d/%s' % (i, path) if i is not None else path
        return 'run-%03d/%s' % (self.i, path)

    def load_q(self, path='reg', i=None, opts=()):
        with closing(h5py.File(self.outfile, 'r')) as f:
            group = f[self._get_path(path, i)]
            with regressor._patch_h5(group):
                self.q.load_weights('whatever')

            try:
                self.scaler_y = regressor._loads(group.attrs['scaler'])
            except KeyError:
                pass

            return {o: group.attrs[o] for o in opts}



    def save(self, group):
        group = group.create_group(None)
        with regressor._patch_h5(group):
            self.q.save_weights('whatever', overwrite=True)

        return group




def run_many(fn, n=20, workers=20, **config):
    seeds = utils.load_dataset('seeds.h5')[:n]

    logger = logging.getLogger('trl')
    logger.setLevel(logging.ERROR)

    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        start_time = time.time()
        futures = {executor.submit(fn, i, int(seed[0] % 2**32), **config): i
                   for i, seed in enumerate(seeds)}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                results[i] = r = future.result()
            except Exception as exc:
                logging.info('%d generated an exception.' % i, exc_info=1)
            else:
                logging.info('Experiment %s completed in %d epochs: %s',
                             i, len(r) -1, r[-1])
                utils.save_dataset(r, 'history.h5', 'history-%d' % i)
        t = time.time() - start_time
        logging.info('Finished in %f (avg: %f)', t, t / len(futures))

    return results


if __name__ == '__main__':
    import glob
    import os
    import signal
    import sys

    signal.signal(signal.SIGINT, handler)

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.chdir(dataset_path)


    #run_many(run, 20)

    n = max((int(f[4:-3]) for f in glob.glob('exp-*.h5')), default=0) + 1
    out_file = 'exp-%03d.h5' % n
    if os.path.exists(out_file):
        raise Exception('nuooo')


    Experiment(9, 0, out_file).run(50)
