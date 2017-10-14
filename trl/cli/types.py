import importlib
import click
import gym

from trl import evaluation, regressor, utils


def handle_index(ctx, value):
    return value.format(i=ctx.meta.get('experiment.index', 0))


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
        value = handle_index(ctx, value)
        if ':' in value:
            return self.load_obj(value)

        return self.convert_not_loadable(value, param, ctx)
    
    def convert_not_loadable(self, value, param, ctx):
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

    def convert_not_loadable(self, value, param, ctx):
        try:
            return utils.load_dataset(value, self.dataset_name)
        except OSError:
            self.fail('Unable to read dataset from %s' % value)


class Regressor(Callable):
    def __init__(self, regressor_name='regressor'):
        self.regressor_name = regressor_name

    def convert_not_loadable(self, value, param, ctx):
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
            self.fail('Unable to load seed from %r' % value)
        except KeyError:
            return self.convert_legacy(value, param, ctx)
        else:
            return int.from_bytes(data, 'big')

    def convert_legacy(self, value, param, ctx):
        try:
            data = utils.load_dataset(value)
            npy_seed, env_seed = data[ctx.meta.get('experiment.index', 0)]
        except Exception:
            self.fail('Unable to load seed from %r' % value)
        else:
            return npy_seed * 256 ** 8 + env_seed


class Path(click.ParamType):
    name = 'path'

    def convert(self, value, param, ctx):
        return handle_index(ctx, value)


# could use env var?
def set_default_output(ctx, param, value):
    if value is not None:
        ctx.meta['default.output'] = handle_index(ctx, value)


def default_output():
    ctx = click.get_current_context()
    return ctx.meta.get('default.output')


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


LOADABLE = Loadable()
CALLABLE = Callable()
DATASET = Dataset()
ENV = EnvParamType()
PATH = Path()
SEED = Seed()


_discounted = lambda e: evaluation.discounted(e.gamma)
METRIC = Metric({
    'discounted': _discounted,
    'dis': _discounted,
    'avg': lambda e: evaluation.average,
})