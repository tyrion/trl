# NOTE: these imports are shortcuts, do not delete
from .commands import cli, experiment, processor
from .logging import LoggingOption
from .types import *


def main():
    import click
    import logging.config

    from ifqi import envs # register ifqi envs
    from trl.algorithms.base import AlgorithmMeta

    from .commands import logger, log_exc
    from .logging import LOGGING

    logging.config.dictConfig(LOGGING)

    # register all the algorithms
    for name, cls in AlgorithmMeta.registry.items():
        experiment.add_command(cls.make_cli(), name)

    try:
        cli(standalone_mode=False)
    except (click.Abort, KeyboardInterrupt):
        logger.error('Interrupt')
    except click.ClickException as exc:
        logger.error(exc.format_message())
    except Exception as exc:
        log_exc(logger, exc)



if __name__ == '__main__':
    main()
