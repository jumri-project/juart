import os

import h5py
import numpy as np

path = os.path.split(__file__)[0]


def jemris_parmaps():
    f = h5py.File(os.path.join(path, "brain80.h5"), mode="r")

    x = f["/sample/data"]
    x = np.array(x)[0, :, :, :]

    x = np.pad(
        x, (((256 - x.shape[0]) // 2,), ((256 - x.shape[1]) // 2,), (0,)), mode="edge"
    )

    x = np.rot90(x, k=2)

    pd = x[:, :, 0]

    support = pd > 0

    r1 = x[:, :, 1] * support
    r2 = x[:, :, 2] * support
    r2s = x[:, :, 3] * support
    fm = x[:, :, 4] / 1000 * support

    return pd, r1, r2, r2s, fm
