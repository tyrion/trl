
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from ifqi import envs
from ifqi.evaluation import evaluation

from trl.evaluation import collect_episodes
from trl import algorithms, regressor, utils

import logging
import logging.config

logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(levelname)s:%(name)s: %(msg)r',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': logging.INFO,
            'formatter': 'default',
        },
    },
    'loggers': {
        'trl': {
            'level': logging.DEBUG,
        },
    },
    'root': {
        'level': logging.INFO,
        'handlers': ['console'],
    },

})


env = envs.LQG1D()
dataset = collect_episodes(env, n=100)

states = np.linspace(-10, 10, 20)
actions = np.linspace(-8, 8, 20)
initial_states = np.array([[1, 2, 5, 7, 10]]).T

ndata = len(dataset)
SA = utils.make_grid(states, actions)
S, A = SA[:, 0], SA[:, 1]




K, cov = env.computeOptimalK(), 0.001
print('Optimal K: {} Covariance S: {}'.format(K, cov))

Q_fun_ = np.vectorize(lambda s, a: env.computeQFunction(s, a, K, cov, 1))
Q_fun = lambda X: Q_fun_(X[:, 0], X[:, 1])
Q_opt = Q_fun(SA)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S, A, Q_opt)


class QPolicy:
    def __init__(self, Q):
        self.Q = Q

    def draw_action(self, states, absorbing=False, evaluation=False):
        v = self.Q(utils.make_grid(states, actions))
        return actions[v.argmax()]

class OptimalPolicy:
    K = env.computeOptimalK()[0][0]

    def draw_action(self, states, absorbing, evaluation=False):
        i = np.abs(actions - self.K*states).argmin()
        #print("states: {} action: {}".format(states, discrete_actions[i]))
        return discrete_actions[i]

def evaluateP(policy, i=initial_states):
    values = evaluation.evaluate_policy(env, policy, initial_states=i)
    print("values (mean {:8.2f},  se {:8.2f})\n steps (mean {:8.2f},  se {:8.2f})".format(*values))
    return values

def evaluateQ(Q, i=initial_states):
    return evaluateP(QPolicy(Q), i)

optimalP = QPolicy(Q_fun)
#optimalP = OptimalPolicy()
_ = evaluateP(optimalP)




class CurveFitQRegressor(regressor.Regressor):

    def fit(self, x, y):
        self.params, pcov = curve_fit(self.Q, x, y, p0=self.params-0.0001)

    def Q(self, sa, b, k):
        s, a = sa[:, 0], sa[:, 1]
        return - b * b * s * a - 0.5 * k * a * a - 0.4* k * s * s

    def predict(self, x):
        return self.Q(x, *self.params)

    def evaluate(self, X=SA):
        Q_hat = self.predict(X)
        Q_opt = Q_fun(X)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], Q_opt)
        ax.scatter(X[:, 0], X[:, 1], Q_hat, c='r', marker='^')

        mae = mean_absolute_error(Q_opt, Q_hat)
        mse = mean_squared_error(Q_opt, Q_hat)

        print("       ( mse {:8.2f}, mae {:8.2f})".format(mse, mae))
        print(" theta (   b {:8.2f},   k {:8.2f})".format(*self.params))

        evaluateQ(self.predict)
        return Q_hat




ACTIVATION = 'sigmoid'
input_dim = 2

model = Sequential()
model.add(Dense(20, input_dim=input_dim, init='uniform', activation=ACTIVATION))
model.add(Dense(2, init='uniform', activation='linear'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
bo = regressor.KerasRegressor(model, input_dim)

q = CurveFitQRegressor(np.array([0.0, 0.0]))
pbo = algorithms.NESPBO(dataset, actions, q, bo, env.gamma, K=1,
             batchSize=10, learningRate=0.01)

pbo.run(100, 1)
pbo.q.evaluate()

plt.show()
