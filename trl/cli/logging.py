import logging
import logging.config

import click

from .types import handle_index


LOGGING_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

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
            'level': 'DEBUG',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'out.log',
            'delay': True,
        },
        'others': {
            'class': 'logging.StreamHandler',
            'level': 'WARNING',
            'formatter': 'default',
        },
    },
    'loggers': {
        'trl.cli': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': False,
        },
        'trl': {
            'level': 'INFO',
            'propagate': False,
            'handlers': ['console'],
        },
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['others'],
    },
}


class MutedOption(click.Option):
    def handle_parse_result(self, ctx, opts, args):
        return None, args


def configure_logging_output(ctx, param, value):
    # Logging to stdout is the default
    if value == '-':
        return
    # if --log-output is not specified
    if value is None:
        # mute logging in subprocesses
        if ctx.meta['experiment.index']:
            value = '/dev/null'
        # log to stdout on single process experiment (or master process)
        else:
            return
    LOGGING['handlers']['file']['filename'] = handle_index(ctx, value)
    LOGGING['loggers']['trl']['handlers'] = ['file']


def configure_logging(ctx, param, value):
    if ctx.resilient_parsing:
        return
    try:
        logging.config.fileConfig(value)
    except:
        logging.config.dictConfig(LOGGING)
    logging.captureWarnings(True)


class LoggingOption(click.Option):

    options = [
        MutedOption(('-v', '--verbose', 'log_v'), count=True, expose_value=False),
        MutedOption(('-q', '--quiet', 'log_q'), count=True, expose_value=False),
    ]

    def __init__(self, logger='trl', default='INFO'):
        super().__init__(('--log-level',), is_eager=True,
                         type=click.Choice(LOGGING_LEVELS))
        self._default = LOGGING_LEVELS.index(default)
        self.logger = logger

    def handle_parse_result(self, ctx, opts, args):
        try:
            lvl = LOGGING_LEVELS.index(opts['log_level'])
        except KeyError:
            log_v, log_q = opts.get('log_v', 0), opts.get('log_q', 0)
            if log_v == 0 and log_q == 0:
                return None, args
            lvl = min(4, max(0, self._default - log_v + log_q))
        except ValueError:
            raise click.UsageError('Invalid log-level %s' % opts['log_level'])

        lvl = LOGGING_LEVELS[lvl]
        ctx.params[self.name] = lvl

        return lvl, args

    def register(self, cmd):
        cmd.params.extend(self.options)
        cmd.params.append(self)