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
pat_gradpbo = r'n(\d{2,3})K(\d\d)inc(\d)ind0ui(\d\d)us(\d\d)b(\d\d\d?)no(inf|\d{3})ul(None|auto|be)'

f = float

perf_time = [('performance', f), ('time', f), ('p_conf', f), ('t_conf', f),
             ('p_std', f), ('t_std', f)]

dtype_fqi = np.dtype([('n', int)] + perf_time)

dtype_pbo = np.dtype([
    ('n', int), ('K', int), ('incremental', bool), ('budget', int),
    ('learning_rate', f),
    ('batch_size', int), ('norm', f)] + perf_time)

dtype_gradpbo = np.dtype([
    ('n', int), ('K', int), ('incremental', bool), ('update_index', int),
    ('update_steps', int),
    ('batch_size', int), ('norm', f), ('update_loss', '<S4')] + perf_time)
dtype_generic = np.dtype([('path', '<S50')] + perf_time)



def experiment_runs(experiment_path, pat='experiment-*.h5'):
    runs = glob.glob(os.path.join(experiment_path, pat))
    n = len(runs)
    data = np.zeros((n,2))
    for i, path in enumerate(runs):
        with closing(h5py.File(path)) as fp:
            time = fp['q'].attrs['time']
            perf = fp['summary'][-1]
            data[i] = (perf, time)
    return data


def average_experiment_runs(experiment_path, pat='experiment-*.h5', std=True):
    data = experiment_runs(experiment_path, pat)
    n = len(data)
    mean = data.mean(axis=0)
    std_ = data.std(axis=0)
    conf = 1.96 * std_ / np.sqrt(n)
    return np.array(([mean, conf, std_] if std else [mean, conf])).ravel()


def postprocess_old(base, pat='(.*)', dtype=dtype_generic):
    return postprocess(base, pat, dtype, False)


def postprocess(base, pat='(.*)', dtype=dtype_generic, std=True):
    experiments = os.listdir(base)
    data = np.recarray((len(experiments),), dtype)
    i = 0
    for name in experiments:
        m = re.match(pat, name)
        if m:
            row = []
            for match, field in zip(m.groups(), dtype.names):
                type = dtype.fields[field][0].type
                match = int(match) if type is np.bool_ else match
                row.append(type(match))

            path = os.path.join(base, name)
            row.extend(average_experiment_runs(path, std=std))
            data[i] = tuple(row)
            i += 1

    data = data[:i]

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
