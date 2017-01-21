import numpy as np


def make_grid(x, y):
    nx = len(x)
    ny = len(y)

    # ensure 2D
    x = x.reshape(nx, -1)
    y = y.reshape(ny, -1)

    x = np.tile(x, (1, ny)).reshape(-1, x.shape[1])
    y = np.tile(y, (nx, 1))
    return np.concatenate((x, y), axis=-1)


def rec_to_array(recarray):
    nrows = len(recarray)
    d = recarray.view(float, np.ndarray)
    return d.reshape((nrows, len(d) // nrows))
