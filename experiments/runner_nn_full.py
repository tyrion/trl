import logging
import logging.config
import os
import sys

os.environ.setdefault('KERAS_BACKEND', 'theano')

import pytest
import theano

floatX = theano.config.floatX
import keras  # this will override theano.config.floatX

# respect theano settings.
keras.backend.set_floatx(floatX)
theano.config.floatX = floatX

import numpy as np
import numdifftools as nd
from theano import tensor as T
import base

from ifqi import envs

from trl import algorithms, regressor, utils
from trl.algorithms import ifqi
from trl.experiment import Experiment
import trl.utils as trlutils
from keras.models import Model
from keras.layers.core import Dense
import matplotlib.pyplot as plt

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
        'level': 'DEBUG',  # debug
        'handlers': ['console'],
    },
}

logging.config.dictConfig(LOGGING)

theta0 = np.array([6., 10.001], dtype=floatX)
epochs = 8


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

def build_nn_3(q):
    inputs = trlutils.t_make_inputs(q.trainable_weights, dtype=floatX)
    last_layer_dim = sum([v.get_value().size for v in q.trainable_weights])
    print(last_layer_dim)
    print(inputs)
    print([el.ndim for el in inputs])
    c = trlutils.k_concat(inputs)
    d1 = Dense(10, init='uniform', activation='sigmoid', name='d1')(c)
    d2 = Dense(last_layer_dim, init='uniform', activation='linear', name='d2')(
        d1)
    o = trlutils.Split(inputs)(d2)
    bo = Model(input=inputs, output=o)
    return regressor.KerasRegressor(bo, len(inputs))


class BaseExperiment(Experiment):
    env_name = 'LQG1D-v0'
    training_episodes = 50
    training_iterations = epochs
    evaluation_episodes = 2
    budget = 1

    def get_q(self):
        return build_nn(2,1,'tanh')
        # return base.CurveFitQRegressor(theta0)

    def get_algorithm_config(self):
        # bo needs to be created here due to seed settings.
        ndim = len(self.q.params)
        if self.algorithm_class in [algorithms.GradPBO, algorithms.NESPBO]:
            self.algorithm_config['bo'] = build_nn_3(self.get_q())
            # self.algorithm_config['bo'] = build_nn(ndim, ndim, "tanh")
        return self.algorithm_config


def experiment(opts, algo_c):
    print(algo_c, file=sys.stderr)
    Ex = type('Ex', (BaseExperiment,), opts)

    def get_experiment(algo):
        e = Ex(algorithm_class=algo, algorithm_config=algo_c.copy(),
               initial_states=None, horizon=50)
        return e

    return get_experiment


class FakeRequest():
    param = None


if __name__ == "__main__":

    incremental, activation, K = True, "tanh", 1
    norm_value, update_every = np.inf, 1

    alg_class = algorithms.GradPBO
    opts, config = [
        # {'np_seed': 6652, 'env_seed': 2897270658018522815},
        {},
        {'incremental': incremental, 'K': K, 'update_index': update_every,
         'update_steps': None, 'norm_value': norm_value,
         'batch_size': 10, 'independent': False}
        if alg_class == algorithms.GradPBO else
        {'update_index': update_every,
         'norm_value': norm_value, 'batch_size': 10}
    ]
    cexp = experiment(opts, config)
    e = cexp(alg_class)
    r1 = e.run()
    alg = e.algorithm
    history = alg.history

    # ##########################################
    # # Evaluate the final solution
    # initial_states = np.array([[1, 2, 5, 7, 10]]).T
    # # values = evaluate_policy(env, alg, initial_states=initial_states)
    # # print('Learned theta: {}'.format(alg.q))
    # # print('Final performance of PBO: {}'.format(values))
    #
    # ##########################################
    # # Some plot
    # #    ks = np.array(history['k']).squeeze()
    # weights = np.array(history['theta']).squeeze()
    # ks = - weights[:, 0] ** 2 / weights[:, 1]
    #
    # plt.figure()
    # plt.title('[train] evaluated weights')
    # plt.scatter(weights[:, 0], weights[:, 1], s=50,
    #             c=np.arange(weights.shape[0]),
    #             cmap='viridis', linewidth='0')
    # plt.xlabel('b')
    # plt.ylabel('k')
    # plt.colorbar()
    # plt.savefig(
    #     'LQG_MLP_evaluated_weights_incremental_{}_activation_{}_steps_{}.png'.format(
    #         incremental, activation, K), bbox_inches='tight')
    #
    # plt.figure()
    # plt.plot(ks)
    # plt.ylim([-5., 5.])
    # plt.xlabel('iteration')
    # plt.ylabel('coefficient of max action (opt ~0.6)')
    # plt.savefig(
    #     'LQG_MLP_max_coeff_incremental_{}_activation_{}_steps_{}.png'.format(
    #         incremental, activation, K), bbox_inches='tight')
    #
    # theta = theta0.copy()
    # L = [np.array([theta])]
    # for i in range(K * 200):
    #     theta = alg.apply_bo(theta)
    #     L.append(np.array(theta))
    #
    # L = np.array(L).squeeze()
    # Kr = - L[:, 0] ** 2 / L[:, 1]
    # print(L.shape)
    #
    # # print(theta)
    # # print('K: {}'.format(q_regressor.get_k(theta)))
    # # alg.q.set_weights(theta)
    # #
    # # values = evaluate_policy(env, alg, initial_states=initial_states)
    # # print('Performance: {}'.format(values))
    # #
    # # print('weights: {}'.format(alg.bo._model.get_weights()))
    #
    # plt.figure()
    # plt.title('Application of Bellman operator')
    # plt.subplot(2, 1, 1)
    # plt.scatter(L[:, 0], L[:, 1])
    # plt.xlabel('b')
    # plt.ylabel('k')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(Kr)
    # plt.ylim([-5., 5.])
    # plt.ylabel('coefficient of max action (opt ~0.6)')
    # plt.savefig(
    #     'LQG_MLP_bpo_application_incremental_{}_activation_{}_steps_{}.png'.format(
    #         incremental, activation, K), bbox_inches='tight')
    #
    # B_i, K_i = np.mgrid[-6:6:40j, -5:35:40j]
    # theta = np.column_stack((B_i.ravel(), K_i.ravel()))
    # theta_p = alg.apply_bo(theta)
    #
    # fig = plt.figure(figsize=(15, 10))
    # Q = plt.quiver(theta[:, 0], theta[:, 1], theta_p[:, 0] - theta[:, 0],
    #                theta_p[:, 1] - theta[:, 1], angles='xy')
    # plt.xlabel('b')
    # plt.ylabel('k')
    # plt.scatter(L[:, 0], L[:, 1], c='b')
    # plt.title(
    #     'Gradient field - Act: {}, Inc: {}'.format(activation, incremental))
    # plt.savefig(
    #     'LQG_MLP_grad_field_incremental_{}_activation_{}_steps_{}.png'.format(
    #         incremental, activation, K), bbox_inches='tight')
    # plt.show()
    # # plt.close('all')
