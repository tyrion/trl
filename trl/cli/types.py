import functools
import importlib

import click
import gym

from trl import evaluation, regressor, utils


def handle_index(ctx, value):
    return value.format(**ctx.meta.get('format.opts', {}))


class ParamType(click.ParamType):

    def is_correct_type(self, value):
        return not isinstance(value, str)

    def __call__(self, value, param, ctx):
        # XXX isinstance check to avoid dumb numpy warning.
        if isinstance(value, str) and value == '.':
            value = ctx.lookup_default(param.name)
            if value is None:
                self.fail('Not specified in config', param)

        return value if self.is_correct_type(value) else \
                super().__call__(value, param, ctx)


class EnvParamType(ParamType):
    name = 'env'

    def convert(self, value, param, ctx):
        try:
            return gym.spec(value)
        except gym.error.Error as exc:
            self.fail(str(exc), param)


class Loadable(ParamType):
    name = 'Loadable'

    def get_metavar(self, param):
        return 'PATH'

    def load_obj(self, path, param):
        module_name, obj_name = path.split(':', 1)
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_name)
        # FIXME probably it would be wise to catch Exception here instead
        except Exception as err:
            args = (path, type(err).__name__)
            self.fail('Failed to import %s (%s)' % args, param)
        return obj

    def convert(self, value, param, ctx):
        value = handle_index(ctx, value)
        if ':' in value:
            return self.load_obj(value, param)

        return self.convert_not_loadable(value, param, ctx)

    def convert_not_loadable(self, value, param, ctx):
        self.fail('Not a valid object reference: %s' % value)


class Callable(Loadable):

    def is_correct_type(self, value):
        return callable(value)

    def load_obj(self, path, param):
        obj = super().load_obj(path, param)
        if not self.is_correct_type(obj):
            self.fail('Loaded object is not callable', param)
        return obj


class Dataset(Loadable):

    def __init__(self, dataset_name='dataset'):
        self.dataset_name = dataset_name

    def convert_not_loadable(self, value, param, ctx):
        try:
            return utils.load_dataset(value, self.dataset_name)
        except OSError:
            self.fail('Unable to read dataset from %s' % value, param)


class Regressor(Loadable):
    def __init__(self, regressor_name='regressor'):
        self.regressor_name = regressor_name

    def is_correct_type(self, value):
        return callable(value)

    def convert(self, value, param, ctx):
        if isinstance(value, dict):
            return self.convert_from_dict(value, param, ctx)

        return super().convert(value, param, ctx)

    def load_obj(self, path, param):
        obj = super().load_obj(path, param)
        if isinstance(obj, dict):
            return self.convert_from_dict(value, param, ctx)

        if callable(obj):
            return obj
        self.fail('Loaded object is not a dict or callable.', param)

    def convert_not_loadable(self, value, param, ctx):
        try:
            regr = regressor.load_regressor(value, self.regressor_name)
        except OSError:
            self.fail('Unable to load regressor from %s' % value, param)
        else:
            return lambda *a, **kw: regr

    def convert_from_dict(self, value, param, ctx):
        fn = regressor.KerasRegressor.from_params
        return functools.partial(fn, **value)


class BORegressor(Regressor):
    def __init__(self, regressor_name='bo'):
        super().__init__(regressor_name)

    def convert_from_dict(self, value, param, ctx):
        return utils.build_bo(**value)


class Metric(Callable):

    def __init__(self, shortcuts=None):
        self.shortcuts = shortcuts or {}

    def convert(self, value, param, ctx):
        try:
            return self.shortcuts[value]
        except KeyError:
            return super().convert(value, param, ctx)


class Seed(Dataset):
    name = 'seed'

    def __init__(self, dataset_name='seed'):
        super().__init__(dataset_name)

    def convert_not_loadable(self, value, param, ctx):
        if value.isdigit():
            return int(value)

        try:
            data = utils.load_dataset(value, 'seed')
        except OSError:
            self.fail('Unable to load seed from %r' % value, param)
        except KeyError:
            return self.convert_legacy(value, param, ctx)
        else:
            return int.from_bytes(data, 'big')

    def convert_legacy(self, value, param, ctx):
        try:
            data = utils.load_dataset(value)
            i = ctx.meta['experiment.index']
            # We subtract one because experiment.index starts from one
            npy_seed, env_seed = data[i-1 if i > 0 else i]
        except Exception:
            self.fail('Unable to load legacy seed from %r' % value, param)
        else:
            return npy_seed * 256 ** 8 + env_seed


class IntOrDataset(Dataset):

    def convert(self, value, param, ctx):
        try:
            return int(value)
        except:
            return super().convert(value, param, ctx)


class Path(ParamType):
    name = 'path'

    def convert(self, value, param, ctx):
        return handle_index(ctx, value)


def default_output():
    """Get the configured global default file path or None"""
    ctx = click.get_current_context()
    return ctx.meta.get('default.output')


def policy(ctx, param, value):
    if value is not None:
        if ':' in value:
            return CALLABLE.load_obj(value, param)
        try:
            regr = regressor.load_regressor(value, 'q')
        except OSError:
            raise click.UsageError('Unable to load policy from %s' % value)
        else:
            return lambda e: evaluation.QPolicy(regr, e.actions)
    else:
        return lambda e: e.policy


LOADABLE = Loadable()
CALLABLE = Callable()
DATASET = Dataset()
ENV = EnvParamType()
PATH = Path()
SEED = Seed()
INT_OR_DATASET = IntOrDataset()
BO_REGRESSOR = BORegressor()

_discounted = lambda e: evaluation.discounted(e.gamma)
METRIC = Metric({
    'discounted': _discounted,
    'dis': _discounted,
    'avg': lambda e: evaluation.average,
})