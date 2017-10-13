import concurrent.futures
import logging

import click

from trl.cli import types
from trl.cli.logging import LoggingOption, configure_logging, \
                            configure_logging_output
from trl.experiment import Experiment


class IndexGroup(click.Group):

    def make_context(self, info_name, args, parent=None, index=None, **extra):
        if parent is None and index is not None:
            parent = click.Context(self, info_name=info_name)
            parent.meta['experiment.index'] = index

        return super().make_context(info_name, args, parent, **extra)


def processor(f):
    def wrapper(*args, **kwargs):
        return lambda exp: f(exp, *args, **kwargs)
    return wrapper


@click.group(chain=True, cls=IndexGroup)
@click.argument('env_spec', metavar='ENV', type=types.ENV)
@click.option('-n', metavar='N', type=int, default=1,
              help='Execute N experiments')
@click.option('-o', '--output', is_eager=True, expose_value=False,
              callback=types.set_default_output)
@click.option('-h', '--horizon', metavar='N', type=int, default=100,
              help='Max number of steps per episode.')
@click.option('-g', '--gamma', metavar='N', type=float)
@click.option('-s', '--seed', type=types.SEED, metavar='SEED',
              help='Specify the random seed.')
@click.option('--save-seed', type=types.PATH, default=types.default_output)
@click.option('-L', '--log-output', metavar='PATH', is_eager=True,
              expose_value=False, callback=configure_logging_output)
@click.option('--log-config', metavar='PATH', is_eager=True,
              expose_value=False, callback=configure_logging)
@click.pass_context
def cli(ctx, log_level='INFO', **config):
    logger = logging.getLogger('')
    logger.setLevel(log_level)
    logger = logging.getLogger('trl')
    logger.setLevel(log_level)

LoggingOption().register(cli)


def _run(i):
    return cli(standalone_mode=False, index=i)


@cli.resultcallback()
def process_result(processors, n, **config):
    logger = logging.getLogger('trl.cli')
    ctx = click.get_current_context()

    if n < 2:
        ctx.meta['experiment.index'] = 0

    # if we are in a subprocess an index has been set
    if 'experiment.index' in ctx.meta:
        return invoke_subcommands(ctx, processors, **config)
    
    with concurrent.futures.ProcessPoolExecutor(n) as executor:
        futures = {
            executor.submit(_run, i): i
            for i in range(n)
        }
        results = {}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                results[i] = r = future.result()
            except Exception as exc:
                logger.info('%d generated an exception.' % i, exc_info=1)
            else:
                logger.info('Experiment %s completed: %s', i, r)


def invoke_subcommands(ctx, processors, **config):
    config = {k: v for k, v in config.items() if v is not None}
    ctx.obj = e = Experiment(**config)
    e.log_config()

    for processor in processors:
        processor(e)


@click.command('interact')
@click.option('-p', '--policy', callback=types.policy, metavar='PATH')
@click.option('-e', '--episodes', default=10)
@click.option('-o', '--output', type=types.PATH, default=types.default_output)
@click.option('-c', '--collect/--no-collect', is_flag=True, default=False)
@click.option('-m', '--metric', 'metrics', type=types.METRIC, multiple=True)
@click.option('-r', '--render/--no-render', is_flag=True, default=False)
@click.option('-s', '--stage', metavar='N', type=int)
@processor
def interact(exp, **config):
    config['metrics'] = [m(exp) for m in config['metrics']]
    return exp.interact(**config)

LoggingOption('trl.evaluation').register(interact)


@click.command('collect')
@processor
def collect(exp, **config):
    config['metrics'] = [m(exp) for m in config['metrics']]
    return exp.collect(**config)

collect.params = [p for p in interact.params if not p.name == 'collect']


@click.command('evaluate')
@processor
def evaluate(exp, **config):
    config['metrics'] = [m(exp) for m in config['metrics']]
    return exp.evaluate(**config)

evaluate.params = collect.params[:]
evaluate.params[3] = click.Option(
    ('-m', '--metric', 'metrics'), type=types.METRIC, multiple=True,
    default=('avg', 'dis'))


cli.add_command(interact)
cli.add_command(collect)
cli.add_command(evaluate)