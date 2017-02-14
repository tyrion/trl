import base
import logging.config

import numpy as np
import theano
from keras.models import Model
from keras.layers.core import Dense

from trl import algorithms, experiment, regressor, utils

from ifqi import envs


logging.config.dictConfig(base.LOGGING)

theta0 = np.array([6., 10.001], dtype=theano.config.floatX)


def build_nn3(q):
    inputs = utils.t_make_inputs(q.trainable_weights, dtype=theano.config.floatX)
    last_layer_dim = sum([v.get_value().size for v in q.trainable_weights])
    #print(last_layer_dim)
    #print(inputs)
    #print([el.ndim for el in inputs])
    c = utils.k_concat(inputs)
    d1 = Dense(10, init='uniform', activation='sigmoid', name='d1')(c)
    d2 = Dense(last_layer_dim, init='uniform', activation='linear', name='d2')(
        d1)
    o = utils.Split(inputs)(d2)
    bo = Model(input=inputs, output=o)
    return regressor.KerasRegressor(bo, None)


class Experiment(experiment.Experiment):
    env_name = 'LQG1D-v0'
    training_episodes = 50
    training_iterations = 8
    evaluation_episodes = 2
    budget = 1
    algorithm_class = algorithms.GradPBO
    algorithm_config = {
        'K': 1,
        'incremental': True,
        'norm_value': 2,
        'update_index': 1,
        'batch_size': 10,
    }

    def get_q(self):
        return base.build_nn(2, 1, 'tanh')

    def get_algorithm_config(self):
        # bo needs to be created here due to seed settings.
        self.algorithm_config['bo'] = build_nn3(self.q)
        return self.algorithm_config


if __name__ == "__main__":
    e = Experiment()
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
