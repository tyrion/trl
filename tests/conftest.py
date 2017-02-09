import logging
import logging.config
import os
import sys

os.environ.setdefault('KERAS_BACKEND', 'theano')

import theano

# Backup floatX
floatX = theano.config.floatX

# this will override theano.config.floatX
import keras

# Restore the correct value of floatX
keras.backend.set_floatx(floatX)
theano.config.floatX = floatX

# importing gym to ensure we ovverride its logging configuration.
import gym
import pytest


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

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('')
