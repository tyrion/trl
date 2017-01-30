#!/usr/bin/env python3
import argparse
import logging
import logging.config
import signal
import warnings

import base


from trl import algorithms
from trl.algorithms import ifqi


if __name__ == '__main__':
    signal.signal(signal.SIGINT, base.handler)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('env_name',
        help='The environment to use. Either from ifqi or gym.')
    parser.add_argument('algorithm',
        choices=['fqi', 'pbo', 'ifqi_fqi', 'ifqi_pbo', 'gradfqi', 'gradpbo'],
        help='The algorithm to run')
    parser.add_argument('-n', '--training-iterations',
        metavar='N', type=int, default=50,
        help='number of training iterations. default is 50.')
    parser.add_argument('-t', '--training-episodes',
        metavar='N', type=int, default=100,
        help='Number of training episodes to collect.')
    parser.add_argument('-e', '--evaluation-episodes',
        metavar='N', type=int,
        help='Number of episodes to use for evaluation.')
    parser.add_argument('-h', '--horizon', type=int, metavar='N',
        help='Max number of steps per episode.')
    parser.add_argument('-b', '--budget', type=int, help='budget', metavar='N')

    io = parser.add_argument_group('io', 'Load/Save')
    io.add_argument('-l', '--load', metavar='FILEPATH', dest='load_path',
        help='Load both the dataset and the Q regressor from FILEPATH')
    io.add_argument('--load-dataset', metavar='FILEPATH',
        help='Load the dataset from FILEPATH', dest='dataset_load_path')
    io.add_argument('-q', '--load-regressor', metavar='FILEPATH', dest='q_load_path',
        help='Load the trained Q regressor from FILEPATH. You can also specify'
             'one of {nn,nn2,curve_fit} instead of FILEPATH.', default='nn2')
    io.add_argument('-o', '--save', metavar='FILEPATH', dest='save_path',
        help='Save both the dataset and the Q regressor to FILEPATH')
    io.add_argument('--save-dataset', metavar='FILEPATH',
        help='Save the dataset to FILEPATH', dest='dataset_save_path')
    io.add_argument('--save-regressor', metavar='FILEPATH', dest='q_save_path',
        help='Save the trained Q regressor to FILEPATH')
    io.add_argument('--save-trace', metavar='FILEPATH', dest='trace_save_path',
        help='Save the evaluation trace to FILEPATH')

    others = parser.add_argument_group('others')
    others.add_argument('-r', '--render', action='store_true',
        help='Render the environment during evaluation.')
    others.add_argument('--timeit', type=int, default=0, metavar='N',
        help='Benchmark algorithm, using N repetitions')
    others.add_argument('-s', '--seeds', type=int,
        nargs='+', default=[None, None], metavar='SEED',
        help='specify the random seeds to be used (gym.env, np.random)')

    log = others.add_mutually_exclusive_group()
    log.add_argument('--log-level', metavar='LEVEL', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set loglevel to LEVEL')
    log.add_argument('-v', '--verbose', action='store_true',
        help='Show more output. Same as --log-level DEBUG')
    log.add_argument('--quiet', action='store_true',
        help='Show less output. Same as --log-level ERROR')

    others.add_argument('--help', action='help',
        help='show this help message and exit')
    args = parser.parse_args()

    if args.verbose:
        args.log_level = logging.DEBUG
    elif args.quiet:
        args.log_level = logging.ERROR
    else:
        args.log_level = getattr(logging, args.log_level)

    base.LOGGING['handlers']['console']['level'] = args.log_level
    logging.config.dictConfig(base.LOGGING)


    from ifqi import envs

    args = dict(vars(args))
    seeds = args.pop('seeds')
    args['np_seed'] = seeds[0]
    if len(seeds) > 1:
        args['env_seed'] = seeds[1]

    args['algorithm_class'] = {
        'fqi': algorithms.FQI,
        'pbo': algorithms.NESPBO,
        'ifqi_fqi': ifqi.FQI,
        'ifqi_pbo': ifqi.PBO,
        'gradfqi': algorithms.GradFQI,
        'gradpbo': algorithms.GradPBO,
    }[args.pop('algorithm')]

    args = {k:v for k,v in args.items() if v is not None}

    base.CLIExperiment(**args).run()
