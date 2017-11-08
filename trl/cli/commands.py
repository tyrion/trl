import collections
import concurrent.futures
import logging
import pickle
import pprint

import click
import hyperopt
import numpy as np

from trl import evaluation, floatX, utils
from trl.cli import types
from trl.cli.logging import LoggingOption, configure_logging, \
                            configure_logging_output, LOGGING
from trl.experiment import Experiment


logger = logging.getLogger('trl.cli')

STATUS_OK = 0
STATUS_FAIL = 1


def configure_defaults(ctx, param, value):
    if value is None:
        return
    if not isinstance(value, dict):
        raise click.UsageError("must be a dict")

    #if ctx.default_map is not None:
    #    value = value.copy()
    #    utils.rec_update(value, ctx.default_map)

    ctx.default_map = value


def run_slave(args, i, **kwargs):
    try:
        return experiment(args, standalone_mode=False, index=i, **kwargs)
    except (click.BadParameter, click.UsageError) as exc:
        # XXX make sure click exceptions are picklable
        exc.ctx = None
        raise
    # FIXME ensure picklable
    # except BaseException as exc:


class _DummyFuture:
    __slots__ = ('fn', 'args', 'kwargs')
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def result(self):
        return self.fn(*self.args, **self.kwargs)


class _DummyExecutor:
    def __init__(self, workers):
        pass

    def submit(self, fn, *args, **kwargs):
        return _DummyFuture(fn, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _dummy_as_completed(futures):
    return futures


class LazyResults:
    def __init__(self, n):
        self.n = n
        self.array = None
        self._index = -1
        self.fails = 0
        self.metrics = []

    def make_array(self, v):
        dtype = [('i', int), ('status', bool)]
        self.metrics = [k for k in v.keys() if not k in {'i', 'status'}]
        dtype.extend(
            (k, floatX if isinstance(v[k], float) else np.array(v[k]).dtype)
            for k in self.metrics
        )
        data = np.zeros(self.n, dtype)
        data['status'] = STATUS_FAIL
        return data

    def add(self, result):
        self._index += 1
        if result['status'] == STATUS_FAIL:
            self.fails += 1
            return

        if self.array is None:
            self.array = self.make_array(result)

        for k, v in result.items():
            self.array[k][self._index] = v


def log_exc(logger, exc, prefix=''):
    name, body = exc.__class__.__name__, str(exc)
    if body:
        logger.error('%s%s: %s.', prefix, name, body)
    else:
        logger.error('%s%s', prefix, name)
    logger.debug('%sexception:', prefix, exc_info=exc)


def opt(*args, **kwargs):
    return click.Option(args, **kwargs)


class Frontend(click.Command):
    allow_extra_args = True
    ignore_unknown_options = True
    allow_interspersed_args = False

    params = [
        opt('-c', '--config',
            type=types.LOADABLE, is_eager=True, expose_value=False,
            callback=configure_defaults),
        opt('--hyperopt',
            type=types.LOADABLE),
        opt('-n',
            metavar='N', type=click.IntRange(1), default=1,
            help='Execute N experiments'),
        opt('-w', '--workers',
            metavar='N', type=click.IntRange(1),
            help='Number of worker processes to spawn when running multiple '
                 'experiments (using -n). By default it is equal to number '
                 'of experiments.'),
        opt('-o', '--output'),
    ]

    def __init__(self):
        super().__init__(name='trl', params=self.params)

    def invoke(self, ctx):
        ctx_ = click.get_current_context()
        assert ctx_ is not None
        configure_logging(ctx, None, None)

        hyperopt_space = ctx.params.pop('hyperopt')
        if hyperopt_space:
            self.run_hyperopt(ctx, hyperopt_space)
        else:
            self.run_experiment(ctx, **ctx.params)

    def run_experiment(self, ctx, *, n=1, workers=None, **extra):
        if workers is None:
            workers = n

        extra['default_map'] = ctx.default_map

        if workers == 1:
            Executor, as_completed = _DummyExecutor, _dummy_as_completed
        else:
            Executor = concurrent.futures.ProcessPoolExecutor
            as_completed = concurrent.futures.as_completed
            # mute logging to stdout in subprocesses
            LOGGING['loggers']['trl']['handlers'] = []
            LOGGING['root']['handlers'] = [] # XXX this is lost

        results = LazyResults(n)
        with Executor(workers) as executor:
            futures = collections.OrderedDict(
                (executor.submit(run_slave, ctx.args, i, **extra), i)
                for i in range(1, n+1))
            for future in as_completed(futures):
                i = futures[future]
                try:
                    r = future.result()
                    # FIXME check r to be a dict ?
                except concurrent.futures.process.BrokenProcessPool:
                    r = {'status': STATUS_FAIL}
                    logger.critical(
                        'Experiment %d was terminated abruptly' % i)
                except (KeyboardInterrupt, click.ClickException,
                        click.Abort) as exc:
                    # Do not catch click exceptions because they are most
                    # probably going to happen on every experiment.
                    raise
                except Exception as exc:
                    r = {'status': STATUS_FAIL}
                    log_exc(logger, exc, 'Experiment %d | ' % i)
                else:
                    # try
                    logger.info('Experiment %s completed: %s', i, r)
                    r['status'] = STATUS_OK

                r['i'] = i
                results.add(r)
        # TODO show info about how many errors?
        return self.postprocess(ctx, results, extra['output'])

    def postprocess(self, ctx, results, output):
        if results.fails == results.n:
            return results, None

        ok_results = results.array[results.array['status'] == STATUS_OK]
        summary = np.zeros(len(results.metrics), {
            'names': ('metric', 'mean', 'std', 'conf'),
            'formats': ('S20', floatX, floatX, floatX)
        })

        sqrt = np.sqrt(results.n)
        logger.info('%20s |       mean |        std |   95%% conf' % 'metric')
        for i, metric in enumerate(results.metrics):
            values = ok_results[metric]
            std = values.std()
            summary[i] = s = (metric, values.mean(), std, 1.96 * std / sqrt)
            logger.info('{:>20s} | {:10.5g} | {:10.5g} | {:10.5g}'.format(*s))

        if output is not None:
            output = output.format(i=0, t=0)
            utils.save_dataset(results.array, output, 'postprocess/results')
            utils.save_dataset(summary, output, 'postprocess/summary')

        return results, summary

    def run_hyperopt(self, ctx, space):
        meta = space.get('_meta', {})
        max_evals = meta.get('max_evals', 10)
        logger.info('Starting HyperOpt (max_evals: %d)', max_evals)

        try:
            trials_path = meta['trials']
            with open(trials_path, 'rb') as fp:
                trials = pickle.load(fp)
        except KeyError:
            trials = None
        except Exception as exc:
            #logging.error('Error while loading trials', exc_info=exc)
            trials = hyperopt.Trials()
        else:
            logger.info('Loaded %d trials from %s', len(trials), trials_path)

        try:
            hyperopt.fmin(
                self.run_hyperopt_experiment, space=space, max_evals=max_evals,
                algo=hyperopt.tpe.suggest, verbose=2, trials=trials,
                return_argmin=0, pass_expr_memo_ctrl=True)
        except (SystemExit, KeyboardInterrupt, click.Abort):
            logger.error('Interrupt')
        except click.UsageError as e:
            logger.error(e.format_message())
            return
        except Exception as exc:
            logger.error("Error during fmin", exc_info=exc)

        logger.info('Finished hyperopt')

        if trials is None or len(trials) < 1:
            return

        try:
            tt = trials._dynamic_trials[-1]
            if tt['result']['status'] not in (hyperopt.STATUS_OK, hyperopt.STATUS_FAIL):
                logger.info('Discarding last trial')
                trials._ids.discard(tt['tid'])
                del trials._dynamic_trials[-1]
                trials.refresh()
        except Exception as exc:
            logger.error('Something went wrong.', exc_info=exc)

        if len(trials) < 1: # should count the successful statuses
            return

        try:
            best = trials.best_trial
        except: # See https://github.com/hyperopt/hyperopt/issues/339
            pass
        else:
            result = best['result']
            logger.info("Best trial (tid: %d, loss: %f, loss_var: %f)", best['tid'],
                        result['loss'], result.get('loss_variance', 0))
            space = hyperopt.space_eval(space, trials.argmin)
            logger.info("Best trial space: \n%s", pprint.pformat(space))
        finally:
            with open(trials_path, 'wb') as fp:
                pickle.dump(trials, fp)


    def run_hyperopt_experiment(self, expr, memo, ctrl):
        ctx = click.get_current_context()
        space = hyperopt.pyll.rec_eval(expr, memo=memo,
                                       print_node_on_error=False)
        tid = ctrl.current_trial['tid']
        logger.info("Starting trial %d", tid)

        if ctx.default_map is not None:
            space = space.copy()
            utils.rec_update(space, ctx.default_map)
        ctx.default_map = space

        results, summary = self.run_experiment(
            ctx, hyperopt_tid=tid, hyperopt_space=space, **ctx.params)

        if summary is None:
            return {'status': hyperopt.STATUS_FAIL}

        # TODO ask for confirmation if results.fails > 0
        try:
            cond = summary['metric'] == np.array('eval.score', 'S')
            score = summary[cond][0]
        except IndexError:
            raise click.UsageError(
                'Forgot to specify "evaluate" as last command?')

        res = {
            'loss': -score['mean'],
            'loss_variance': score['std'] ** 2,
            'status': hyperopt.STATUS_OK,
        }
        logger.info('Trial %d completed: %s' % (tid, res))
        return res


    def get_help(self, ctx):
        return cli.get_help(ctx)

cli = Frontend()


class Group(click.Group):

    def make_context(self, info_name, args, parent=None, index=0, output=None,
                     hyperopt_space=None, hyperopt_tid=None, **extra):
        if parent is None:
            parent = click.Context(self, info_name='')
            parent.meta['hyperopt.space'] = hyperopt_space
            parent.meta['hyperopt.tid'] = hyperopt_tid
            parent.meta['experiment.index'] = index
            # FIXME the or is a hack, loading files lazily is a solution
            parent.meta['format.opts'] = opts = {
                'i': index,
                't': hyperopt_tid or 0
            }
            if output is not None:
                parent.meta['default.output'] = output.format(**opts)
        return super().make_context(info_name, args, parent, **extra)


@click.group(chain=True, cls=Group)
@click.argument('env_spec', metavar='ENV', type=types.ENV)
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
def experiment(ctx, log_level='INFO', **config):
    logger = logging.getLogger('')
    logger.setLevel(log_level)
    logger = logging.getLogger('trl')
    logger.setLevel(log_level)

LoggingOption().register(experiment)


@experiment.resultcallback()
def process_result(processors, **config):
    ctx = click.get_current_context()
    return invoke_subcommands(ctx, processors, **config)


def invoke_subcommands(ctx, processors, **config):
    config = {k: v for k, v in config.items() if v is not None}
    ctx.obj = e = Experiment(**config)
    e.log_config()

    for processor in processors:
        processor(e)

    return e.metrics


def processor(f):
    def wrapper(*args, **kwargs):
        return lambda exp: f(exp, *args, **kwargs)
    return wrapper


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


experiment.add_command(interact)
experiment.add_command(collect)
experiment.add_command(evaluate)