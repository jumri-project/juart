import numpy as np


def zpad(x, s, value=0):

    m = np.array(x.shape)
    s = np.array(s)

    assert (m.size == s.size)
    assert (np.all(s >= m))

    y = np.zeros(s, dtype=x.dtype) + value

    idx = [slice(int(np.floor(s[n] / 2) + np.ceil(-m[n] / 2)),
                 int(np.floor(s[n] / 2) + np.ceil(+m[n] / 2)))
           for n in range(len(s))]

    y[tuple(idx)] = x

    return y
