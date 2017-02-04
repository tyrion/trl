from ifqi import envs  # registra gli env di ifqi
from trl import utils, evaluation, algorithms, regressor
import gym
import numpy as np
import theano
import logging
import theano
import theano.tensor as T
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras

theano.config.floatX = "float32"
keras.backend.set_floatx(theano.config.floatX)


class CurveFitQRegressor(regressor.Regressor):
    def __init__(self, params):
        self._params = p = theano.shared(np.array(params, dtype=theano.config.floatX),
                                         'params', allow_downcast=True)
        self.sa = sa = T.matrix('sa', dtype=theano.config.floatX)
        self.s, self.a = sa[:, 0], sa[:, 1]
        self.s.name = 's'
        self.a.name = 'a'
        self.j = self.model(sa, p)
        self.inputs = [sa]
        self.outputs = [self.j]
        # TODO check if it is faster like this, or with normal predict
        self.predict = theano.function([sa], self.j)

    @property
    def trainable_weights(self):
        return [self._params]

    @property
    def params(self):
        return self._params.get_value()

    @params.setter
    def params(self, params):
        self._params.set_value(params)

    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params - 0.0001)

    def Q(self, sa, b, k):
        return self.model(sa, [b, k])

    def model(self, sa, theta):
        s = sa[:, 0]
        a = sa[:, 1]
        b = theta[0]
        k = theta[1]
        return - b * b * s * a - 0.5 * k * a * a - 0.4 * k * s * s

    def predict(self, x):
        pass

    def get_k(self, omega):
        b = omega[0]
        k = omega[1]
        return - b * b / k


def build_nn(input_dim=2, output_dim=2, activation='tanh'):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import callbacks

    model = Sequential()
    model.add(Dense(2, input_dim=input_dim, init='uniform',
                    activation=activation))
    model.add(Dense(output_dim, init='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    print(model.get_weights())
    return regressor.KerasRegressor(model, input_dim)


def ifqi_bo_q(ACTIVATION="tanh"):
    ### Q REGRESSOR ##########################
    class LQG_Q(object):
        def model(self, s, a, omega):
            b = omega[:, 0]
            k = omega[:, 1]
            q = - b * b * s * a - 0.5 * k * a * a - 0.4 * k * s * s
            return q.ravel()

        def n_params(self):
            return 2

        def get_k(self, omega):
            b = omega[:, 0]
            k = omega[:, 1]
            return - b * b / k

        def name(self):
            return "R1"

    q_regressor = LQG_Q()
    ##########################################

    ### F_RHO REGRESSOR ######################
    n_q_regressors_weights = q_regressor.n_params()
    Sequential.n_inputs = lambda self: n_q_regressors_weights

    def _model_evaluation(self, theta):
        inv = theta
        for el in self.flattened_layers:
            # print(el)
            inv = el(inv)
        return inv

    Sequential.model = _model_evaluation
    rho_regressor = Sequential()
    rho_regressor.add(Dense(2, input_dim=n_q_regressors_weights, init='uniform', activation=ACTIVATION))
    rho_regressor.add(Dense(n_q_regressors_weights, init='uniform', activation='linear'))
    rho_regressor.compile(loss='mse', optimizer='rmsprop')
    print(rho_regressor.get_weights())

    return q_regressor, rho_regressor


class AlgSettings(object):
    def __init__(self, dataset, actions, gamma, q, state_dim, action_dim):
        self.dataset = dataset
        self.actions = actions
        self.gamma = gamma
        self.q = q
        self.state_dim = state_dim
        self.action_dim = action_dim


if __name__ == "__main__":
    env = gym.make('LQG1D-v0')
    import ifqi

    state_dim, action_dim, reward_dim = ifqi.envs.get_space_info(env)

    np_seed = 1312312
    env_seed = 2236864
    np.random.seed(np_seed)
    env.seed(env_seed)

    episodes = 10
    interaction = evaluation.Interact(env, episodes, horizon=10, collect=True)
    interaction.interact()
    dataset = interaction.dataset  # il dataset è qui dentro
    discrete_actions = np.linspace(-8, 8, 20)

    theta0 = np.array([6., 10.001], dtype=theano.config.floatX)
    q = CurveFitQRegressor(theta0)
    bo = build_nn(input_dim=2, output_dim=2, activation="tanh")

    norm_value, incremental, K = (1, True, 1)
    nbep, batch_size, update_every = 2, 1, 1
    update_step = None

    sts = AlgSettings(dataset, discrete_actions, env.gamma, q, 1, 1)

    alg = algorithms.GradPBO(sts, bo, K=K, optimizer="adam", batch_size=batch_size,
                             norm_value=norm_value, update_index=update_every, update_steps=update_step,
                             incremental=incremental, independent=False)

    alg.run(n=nbep)
    weights = np.array(alg.history['theta']).squeeze()
    rhos = alg.history['rho']

    np.random.seed(np_seed)

    import ifqi.algorithms.pbo.gradpbo as ifqipbo

    ifq, ifbo = ifqi_bo_q()
    pbo = ifqipbo.GradPBO(bellman_model=ifbo,
                          q_model=ifq,
                          steps_ahead=K,
                          discrete_actions=discrete_actions,
                          gamma=env.gamma,
                          optimizer="adam",
                          state_dim=state_dim,
                          action_dim=action_dim,
                          incremental=incremental,
                          update_theta_every=update_every,
                          steps_per_theta_update=update_step,
                          norm_value=norm_value,
                          verbose=0,
                          independent=False)

    flatds = utils.rec_to_array(dataset[['state', 'action', 'reward', 'next_state', 'absorbing', 'done']])

    from ifqi.evaluation.utils import split_dataset

    state, actions, reward, next_states = split_dataset(flatds,
                                                        state_dim=state_dim,
                                                        action_dim=action_dim,
                                                        reward_dim=reward_dim)

    theta0 = theta0.reshape(1, -1)
    history = pbo.fit(state, actions, next_states, reward, theta0,
                      batch_size=batch_size, nb_epoch=nbep, verbose=0)
    ifqiweights = np.array(history.hist['theta']).squeeze()

    XX = np.column_stack((weights - ifqiweights, weights, ifqiweights))
    for i in range(XX.shape[0]):
        el = XX[i]
        if not np.isfinite(el[0]) or el[0] > 1e-10:
            print("iteration: [errore theta] | [theta TRL] | [theta IFQI]")
            print("{}: {}".format(i - 1, XX[i - 1]))
            print("{}: {}".format(i, el))
            print("NN weights")
            print("in TRL")
            print(alg.history['rho'][i])
            print(alg.history['rho2'][i])
            print("in IFQI")
            print(history.hist['rho'][i])
            break
    assert np.allclose(weights, ifqiweights)
