from contextlib import closing
import re
import os
import glob

import h5py
import numpy as np
import pandas as pd
from pandas import ExcelWriter

from trl import utils


pat_fqi = r'n(\d\d)'
pat_pbo = r'n(\d\d)K(\d\d)inc(\d)B(\d\d)l(\d\.\d\d)b(\d\d\d)no(inf|\d\d\d)'
pat_gradpbo = r'n(\d\d)K(\d\d)inc(\d)ind0ui(\d\d)us\d\db(\d\d\d?)no(inf|\d\d\d)'

f = float

dtype_fqi = [('n', int), ('performance', f), ('time', f),
    ('p_conf', f), ('t_conf', f)]

dtype_pbo = [
    ('n', int), ('K', int), ('incremental', bool), ('budget', int),
    ('learning_rate', f),
    ('batch_size', int), ('norm', f), ('performance', f), ('time', f),
    ('p_conf', f), ('t_conf', f)]

dtype_gradpbo = [
    ('n', int), ('K', int), ('incremental', bool), ('update_index', int),
    ('batch_size', int), ('norm', f), ('performance', f), ('time', f),
    ('p_conf', f), ('t_conf', f)]


def average_experiment_runs(experiment_path):
    runs = glob.glob(os.path.join(experiment_path, 'experiment-*.h5'))
    n = len(runs)
    data = np.zeros((n,2))
    for i, path in enumerate(runs):
        with closing(h5py.File(path)) as fp:
            time = fp['q'].attrs['time']
            perf = fp['summary'][-1]
            data[i] = (perf, time)

    mean = data.mean(axis=0)
    conf = 1.96 * data.std(axis=0) / np.sqrt(n)
    return np.array([mean, conf]).ravel()

def postprocess(base, pat, dtype):
    experiments = os.listdir(base)
    data = np.recarray((len(experiments),), dtype)
    for i, name in enumerate(experiments):
        m = re.match(pat, name)
        if m:
            row = [t(x) for x, (_, t) in zip(m.groups(), dtype)]

            path = os.path.join(base, name)
            row.extend(average_experiment_runs(path))
            data[i] = tuple(row)

    utils.save_dataset(data, 'results-{}.h5'.format(base))

    df = pd.DataFrame(data)
    df.to_csv('results-{}.csv'.format(base))
    return df, data


if __name__ == '__main__':
    import sys
    base = sys.argv[1]
    pat = globals()['pat_{}'.format(base)]
    dtype = globals()['dtype_{}'.format(base)]

    postprocess(base, pat, dtype)
