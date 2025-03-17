from typing import Tuple, Union

import numpy as np
import pandas as pd

from . import utils as ut


def emptyimage(
    res: Union[int, np.ndarray], fov: Union[float, np.ndarray], ndim=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates the pixel locations inside
    a empty image with Res matrix size inside the fov.

    Parameters
    ----------
    res : int or np.ndarray or list.
        Matrix size of the image.
    fov : float or np.ndarray or list.
        Field of view [m].
    ndim : int or None
        Number of dimensions. Defaults to None.

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        loc : Raveled pixel locations inside the fov of shape (Nsamples, ndim) [m].
        meshgrid : Pixel locations meshgrid inside the fov of shape res [m].
    """
    res, ndim_res = ut._checkdim(res, ndim, "resolution")
    fov, ndim_fov = ut._checkdim(fov, ndim, "fov")

    if len(res) != len(fov):
        raise ValueError(
            f"Dim of resolution {res.shape} does not match dim of fov {fov.shape}"
        )

    if ndim_res != ndim_fov:
        raise ValueError(
            f"Dim of resolution {ndim_res} does not match dim of fov {ndim_fov}"
        )
    else:
        ndim = ndim_res

    grid_dims = []

    for r, f in zip(res, fov):
        grid_dims.append(np.linspace(-f / 2, f / 2, r))

    # create meshgrid (dim0, dim1, ....)
    meshgrid = np.meshgrid(*grid_dims, indexing="ij")

    # Initialize matrix (samples, dims)
    loc = np.column_stack(tuple([arg.flatten() for arg in meshgrid]))

    return loc, meshgrid


def ellipsemask(input: np.ndarray, params: pd.DataFrame) -> np.ndarray:
    """Returns a mask which elements of the input
    are inside a ellipsoid.

    Parameters
    ----------
    input : np.ndarray
        Locations in the field of view (Nsamples, Ndim) [m].
    params : pd.DataFrame
        Shape parameters of ellipsoid in [m] and [degree].
        Keys 3D: [center_x, center_y, center_z, axis_a, axis_b, axis_c, angle]

    Returns
    -------
    np.ndarray
        Mask which samples of "input" are inside the ellipsoid.
    """
    Nsamples, Ndim = input.shape

    ang = np.deg2rad(params.angle)

    # extract centers, axes and rotation angle
    if Ndim == 2:
        axes = np.array([params.axis_a, params.axis_b])
        cent = np.array([params.center_x, params.center_y])

        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    elif Ndim == 3:
        # define some temp variables for easier calculation
        axes = np.array([params.axis_a, params.axis_b, params.axis_c])
        cent = np.array([params.center_x, params.center_y, params.center_z])

        # rotation of ellipsoid
        R = np.array(
            [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
        )

    # transform ellipsoid to sphere
    sphere = (((input - cent) @ R) / axes) ** 2

    mask = np.sum(sphere, axis=1) <= 1

    return mask
