import numpy as np


def make_grid(x, y):
    m = np.meshgrid(x, y, copy=False, indexing='ij')
    return np.vstack(m).reshape(2, -1).T


def rec_to_array(recarray):
    nrows = len(recarray)
    d = recarray.view(float, np.ndarray)
    return d.reshape((nrows, len(d) // nrows))
