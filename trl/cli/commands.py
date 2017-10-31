import concurrent.futures
import logging
import pickle

import click
import hyperopt
import numpy as np

from trl import evaluation, utils
from trl.cli import types
from trl.cli.logging import LoggingOption, configure_logging, \
                            configure_logging_output
from trl.experiment import Experiment


logger = logging.getLogger('trl.cli')


class Group(click.Group):

    def make_context(self, info_name, args, parent=None, index=None,
                     hyperopt_space=None, hyperopt_tid=None, **extra):
        if parent is None:
            parent = click.Context(self, info_name='')
            parent.meta['hyperopt.space'] = hyperopt_space
            parent.meta['hyperopt.tid'] = hyperopt_tid
            parent.meta['experiment.index'] = index
            # FIXME the or is a hack, loading files lazily is a solution
            parent.meta['format.opts'] = {
                'i': index or 0,
                't': hyperopt_tid or 0
            }
        return super().make_context(info_name, args, parent, **extra)


def processor(f):
    def wrapper(*args, **kwargs):
        return lambda exp: f(exp, *args, **kwargs)
    return wrapper


def configure_defaults(ctx, param, value):
    if value is None:
        return
    if not isinstance(value, dict):
        raise click.UsageError("must be a dict")

    if ctx.default_map is not None:
        value = value.copy()
        utils.rec_update(value, ctx.default_map)

    ctx.default_map = value


def configure_hyperopt(ctx, param, value):
    if value is None or ctx.resilient_parsing:
        return

    if ctx.meta['hyperopt.space'] is not None:
        return True

    _meta = value.get('_meta', {})
    max_evals = _meta.get('max_evals', 10)

    try:
        trials_path = _meta['trials']
        with open(trials_path, 'rb') as fp:
            trials = pickle.load(fp)
        print(len(trials))
    except KeyError:
        trials = None
    except Exception as exc:
        #logging.error('Error while loading trials', exc_info=exc)
        trials = hyperopt.Trials()
    else:
        print('Loaded trials from', trials_path)

    try:
        hyperopt.fmin(hyperopt_run_master, space=value, max_evals=max_evals,
                        algo=hyperopt.tpe.suggest, verbose=2, trials=trials,
                        return_argmin=0, pass_expr_memo_ctrl=True)
    except (SystemExit, KeyboardInterrupt, click.Abort):
        print('Interrupt')
    except Exception as exc:
        logging.error("Error during fmin", exc_info=exc)

    print('Finished hyperopt')

    if trials is None or len(trials) < 1:
        ctx.exit()

    tt = trials._dynamic_trials[-1]
    if tt['result']['status'] not in (hyperopt.STATUS_OK, hyperopt.STATUS_FAIL):
        print('discarding last trial')
        trials._ids.discard(tt['tid'])
        del trials._dynamic_trials[-1]
        trials.refresh()

    if len(trials) < 1: # should count the successful statuses
        ctx.exit()

    print(trials.argmin)
    print(trials.best_trial['result'])

    with open(trials_path, 'wb') as fp:
        pickle.dump(trials, fp)

    ctx.exit()


@click.group(chain=True, cls=Group)
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
@click.option('-c', '--config', type=types.LOADABLE, is_eager=True,
              expose_value=False, callback=configure_defaults)
@click.option('--hyperopt', type=types.LOADABLE, is_eager=True,
              callback=configure_hyperopt)
@click.pass_context
def cli(ctx, log_level='INFO', **config):
    logger = logging.getLogger('')
    logger.setLevel(log_level)
    logger = logging.getLogger('trl')
    logger.setLevel(log_level)

LoggingOption().register(cli)


@cli.resultcallback()
def process_result(processors, n, **config):
    ctx = click.get_current_context()

    run_kwargs = {
        'hyperopt_space': ctx.meta['hyperopt.space'],
        'default_map': ctx.meta['hyperopt.space'],
        'hyperopt_tid': ctx.meta['hyperopt.tid']
    } if config['hyperopt'] else {}

    # if we are in a subprocess an index has been set
    if ctx.meta['experiment.index'] is not None:
        return invoke_subcommands(ctx, processors, **config)

    results = {}
    # if we need to run just one experiment do not start the multiprocessing
    # machinery.
    if n < 2:
        ctx.meta['experiment.index'] = 0
        # FIXME duplicated code from run_slave
        exp, res = invoke_subcommands(ctx, processors, **config)
        results[0] = res.summary \
                     if isinstance(res, evaluation.Interaction) else None
    else:
        with concurrent.futures.ProcessPoolExecutor(n) as executor:
            futures = {
                executor.submit(run_slave, i, **run_kwargs): i
                for i in range(n)
            }
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results[i] = r = future.result()
                except Exception as exc:
                    logger.info('%d generated an exception.' % i, exc_info=1)
                else:
                    logger.info('Experiment %s completed: %s', i, r)

    return postprocess(results)


def postprocess(results):
    # TODO use a named array for summary
    # Results (mean std 95%-conf): avgJ (0.123123 123123 123123), time (0.123123  123123 123123)
    n = len(results)
    try:
        results = np.vstack(results[i] for i in range(n)) # preserve order
        mean = results.mean(0)
        std = results.std(0)
        conf = 1.96 * std / np.sqrt(n)
        summary = np.stack((mean, std, conf))
    except Exception as e:
        logger.error("error", exc_info=e)
        summary = None
    else:
        logger.info("Summary of results of %d experiments:", n)
        logger.info("|     mean: %s", mean)
        logger.info("|      std: %s", std)
        logger.info("| 95%% conf: %s", conf)

    return (results, summary)


def invoke_subcommands(ctx, processors, **config):
    config = {k: v for k, v in config.items() if v is not None}
    ctx.obj = e = Experiment(**config)
    e.log_config()

    result = None
    for processor in processors:
        result = processor(e)

    return (e, result)


def run_slave(i, **kwargs):
    exp, res = cli(standalone_mode=False, index=i, **kwargs)
    return res.summary if isinstance(res, evaluation.Interaction) else None


def hyperopt_run_master(expr, memo, ctrl):
    space = hyperopt.pyll.rec_eval(expr, memo=memo, print_node_on_error=False)
    tid = ctrl.current_trial['tid']
    try:
        results, summary = cli(standalone_mode=False, default_map=space,
                               hyperopt_tid=tid, hyperopt_space=space)
    # For some reason click swaps KeyboardInterrupt for click.Abort so we have
    # to catch it and re-raise it because it inherits from Exception.
    except click.Abort:
        raise
    except Exception as exc:
        logging.error("Error during trial", exc_info=exc)
        return {'status': hyperopt.STATUS_FAIL}
    else:
        if summary is None or len(summary[0]) < 2: # XXX can this be detected earlier?
            raise click.UsageError('Forgot to specify "evaluate" as last command?')
        res = {
            'loss': -summary[0][1],
            'loss_variance': summary[1][1] ** 2,
            'status': hyperopt.STATUS_OK,
            # 'hash': name,
        }
        print('Trial %d completed: %s' % (tid, res))
        return res


@click.command('interact')
@click.option('-p', '--policy', callback=types.policy, metavar='PATH')
@click.option('-e', '--episodes', default=10, type=types.INT_OR_DATASET)
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