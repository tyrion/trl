import os
import theano

os.environ.setdefault('KERAS_BACKEND', 'theano')
floatX = theano.config.floatX
import keras # this will override theano.config.floatX

# respect theano settings.
keras.backend.set_floatx(floatX)
theano.config.floatX = floatX