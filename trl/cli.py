import importlib
import logging
import logging.config
import os

import click
import gym
import numpy as np

from trl import evaluation, regressor, utils
from trl.evaluation import Interaction
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
            'level': 'DEBUG',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'out.log',
        },
        'others': {
            'class': 'logging.StreamHandler',
            'level': 'WARNING',
            'formatter': 'default',
        },
    },
    'loggers': {
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

logger = logging.getLogger(__name__)


class EnvParamType(click.ParamType):
    name = 'env'

    def convert(self, value, param, ctx):
        try:
            return gym.spec(value)
        except gym.error.Error as exc:
            raise click.BadParameter(str(exc))


class Loadable(click.ParamType):
    name = 'Loadable'

    def get_metavar(self, param):
        return 'PATH'

    def load_obj(self, path):
        module_name, obj_name = path.split(':', 1)
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_name)
        except (AttributeError, ImportError):
            self.fail('Failed to load %s' % path)
        return obj

    def convert(self, value, param, ctx):
        if ':' in value:
            return self.load_obj(value)

        self.fail('Not a valid object reference: %s' % value)


class Callable(Loadable):

    def load_obj(self, path):
        obj = super().load_obj(path)
        if not callable(obj):
            self.fail('Loaded object is not callable')
        return obj


class Dataset(Loadable):

    def __init__(self, dataset_name='dataset'):
        self.dataset_name = dataset_name

    def convert(self, value, param, ctx):
        if ':' in value:
            return self.load_obj(value)

        try:
            return utils.load_dataset(value, self.dataset_name)
        except OSError:
            self.fail('Unable to read dataset from %s' % value)


class Regressor(Callable):
    def __init__(self, regressor_name='regressor'):
        self.regressor_name = regressor_name

    def convert(self, value, param, ctx):
        if ':' in value:
            return self.load_obj(value)

        try:
            regr = regressor.load_regressor(value, self.regressor_name)
        except OSError:
            self.fail('Unable to load regressor from %s' % value)
        else:
            return lambda *a, **kw: regr


class Metric(Callable):

    def __init__(self, shortcuts=None):
        self.shortcuts = shortcuts or {}

    def convert(self, value, param, ctx):
        try:
            return self.shortcuts[value](ctx.obj)
        except KeyError:
            return super().convert(value, param, ctx)


def policy(ctx, param, value):
    if value is not None:
        if ':' in value:
            return CALLABLE.load_obj(value)
        try:
            regr = regressor.load_regressor(value, 'q')
        except OSError:
            raise click.UsageError('Unable to load policy from %s' % value)
        else:
            return lambda e: evaluation.QPolicy(regr, e.actions)
    else:
        return lambda e: e.policy


class LoadablePath(Loadable, click.Path):

    def convert(self, value, param, ctx):
        if ':' in value:
            return self.load_obj(value)
        return super().convert(value, param, ctx)


LOADABLE = Loadable()
CALLABLE = Callable()
DATASET = Dataset()
ENV = EnvParamType()

_discounted = lambda e: evaluation.discounted(e.gamma)
METRIC = Metric({
    'discounted': _discounted,
    'dis': _discounted,
    'avg': lambda e: evaluation.average,
})

LOGGING_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

class MutedOption(click.Option):
    def handle_parse_result(self, ctx, opts, args):
        return None, args


def configure_logging_output(ctx, param, value):
    if value is not None and value != '-':
        LOGGING['handlers']['file']['filename'] = value
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
        super().__init__(('--log-level',), is_eager=True, expose_value=False,
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
        logger = logging.getLogger(self.logger)
        logger.setLevel(lvl)

        return lvl, args

    def register(self, cmd):
        cmd.params.extend(self.options)
        cmd.params.append(self)


def processor(f):

    def wrapper(*args, **kwargs):
        return lambda exp: f(exp, *args, **kwargs)

    return wrapper




@click.group(chain=True)
@click.argument('env_spec', metavar='ENV', type=ENV)
@click.option('-h', '--horizon', metavar='N', type=int, default=100,
              help='Max number of steps per episode.')
@click.option('-g', '--gamma', metavar='N', type=float, default=0.99)
@click.option('-s', '--seed', type=int, metavar='SEED',
              help='Specify the random seed.')
@click.option('--log-output', metavar='PATH', is_eager=True,
              expose_value=False, callback=configure_logging_output)
@click.option('--log-config', metavar='PATH', is_eager=True,
              expose_value=False, callback=configure_logging)
@click.pass_context
def cli(ctx, **config):
    ctx = click.get_current_context()
    pass

LoggingOption().register(cli)


@cli.resultcallback()
def process_result(processors, **config):
    ctx = click.get_current_context()
    config = {k: v for k, v in config.items() if v is not None}
    ctx.obj = e = Experiment(**config)
    e.log_config()

    for processor in processors:
        processor(e)


@click.command('interact')
@click.option('-p', '--policy', callback=policy, metavar='PATH')
@click.option('-e', '--episodes', default=10)
@click.option('-o', '--output') # filepath
@click.option('-c', '--collect/--no-collect', is_flag=True, default=False)
@click.option('-m', '--metric', 'metrics', type=METRIC, multiple=True)
@click.option('-r', '--render/--no-render', is_flag=True, default=False)
@click.option('-s', '--stage', metavar='N', type=int)
@processor
def interact(exp, **conf):
    return exp.interact(**conf)

LoggingOption('trl.evaluation').register(interact)


@click.command('collect')
@processor
def collect(exp, **config):
    return exp.collect(**config)
collect.params = [p for p in interact.params if not p.name == 'collect']


@click.command('evaluate')
@processor
def evaluate(exp, **config):
    return exp.evaluate(**config)

evaluate.params = collect.params[:]
evaluate.params[3] = click.Option(
    ('-m', '--metric', 'metrics'), type=METRIC, multiple=True,
    default=('avg', 'dis'))


cli.add_command(interact)
cli.add_command(collect)
cli.add_command(evaluate)


def main():
    from ifqi import envs
    from trl.algorithms.base import AlgorithmMeta

    for name, cls in AlgorithmMeta.registry.items():
        cli.add_command(cls.make_cli(), name)
    cli()


if __name__ == '__main__':
    main()
