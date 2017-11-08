# trl LQG1D-v0 collect fqi evaluate gradpbo evaluate

# trl -n 20 LQG1D-v0 collect -o data-{}.h5
# trl -n 20 LQG1D-v0 collect fqi evaluate

# trl -n 20 LQG1D-v0 fqi -d data-{}.h5 -i 20 base:curve_fit
# trl -n 20 LQG1D-v0 fqi -d data-{}.h5 -i 40 base:curve_fit
from .commands import processor, cli, experiment
from .logging import LoggingOption
from .types import *


def main():
    from ifqi import envs
    from trl.algorithms.base import AlgorithmMeta

    for name, cls in AlgorithmMeta.registry.items():
        experiment.add_command(cls.make_cli(), name)
    cli()


if __name__ == '__main__':
    main()
