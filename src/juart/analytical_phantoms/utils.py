from typing import Union

import numpy as np
import pandas as pd


def _checkdim(
    input: Union[float, int, list, np.ndarray, tuple],
    ndim: Union[float, int, None],
    inputstr: str,
):
    """Checks if input variable (e.g. resolution or fov) is an scalar or list or array.
    Returns an array of shape (ndim,) with the elements
    of input and the number of dimensions ndim.

    If input is a scalar and ndim is not 2 or 3, Value error is raised.
    If input is a scalar and ndim is 2 or 3,
    an array of shape (ndim) is returned with input as each element.
    If input is a list or array, ndim is changed to the length of input.

    Parameters
    ----------
    input : float or int or list or ndarray or tuple
        Input variable.
    ndim : float or int or None
        Number of dimensions.
    inputstr : str
        String to identify the input variable

    Returns
    -------
    output : np.ndarray
        Modified input as array of shape ndim
    ndim : int
        Number of dimensions.
    """
    if isinstance(input, (float, int)):  # Check if fov is a scalar
        if ndim is None or ndim not in [2, 3]:
            raise ValueError(
                f"If {inputstr} is a scalar.",
                f"ndim should be set to 2 or 3 but is {ndim}.",
            )
        else:
            output = [input] * ndim

    elif isinstance(input, (np.ndarray, list, tuple)) and len(input) in [
        2,
        3,
    ]:  # Check if fov is an array-like structure with valid ndimensions
        ndim = len(input)
        output = input
    else:
        raise ValueError(f"Invalid {inputstr} parameter {input}.")

    output = np.array(output)
    ndim = int(ndim)

    return output, ndim


def createCartImgGridLocation(
    grid_size: Union[int, np.ndarray],
    fov: Union[float, np.ndarray],
    ndim=None,
    format="grid",
):
    """Creates a cartesian grid in image space from [-fov/2, +fov/2] in real
    si units locations r [m].

    Parameters
    ----------
    grid_size : float or np.ndarray
        Grid size of the cartesian grid.
    fov : float, np.ndarray
        Field of view.
    ndim : int, optional
        Number of dimensions in which grid is created, by default None
    format : str, optional
        'grid' -> returns meshgrid of size (Ndim, *grid_size)
        'vec' -> returns matrix of shape (Nsamples, Ndim) as vector
                of grid location for all grid points
        by default 'grid'

    Returns
    -------
    np.meshgrid : if format=='grid'
    np.ndarray : if format='vec'
    """

    grid_size, ndim = _checkdim(grid_size, ndim, "gridsize")
    fov, ndim = _checkdim(fov, ndim, "fov")

    grid_ticks = []

    for g, f in zip(grid_size, fov):
        dr = f / g
        grid_ticks.append(np.linspace(-f / 2, f / 2, g, endpoint=True) + dr / 2)

    mesh_locs = np.meshgrid(*grid_ticks)

    if format == "grid":
        return mesh_locs

    elif format == "vec":
        return np.column_stack([ml.ravel() for ml in mesh_locs])


def createCartKspaceGridLocation(
    grid_size: Union[int, np.ndarray],
    fov: Union[float, np.ndarray],
    ndim=None,
    format="grid",
):
    """Creates a cartesian grid in kspace space from [-kmax, +kmax]
    with kmax = fov * grid_size / 2 in si units locations 1/r [m].

    Parameters
    ----------
    grid_size : float or np.ndarray
        Grid size of the cartesian grid.
    fov : float, np.ndarray
        Field of view.
    ndim : int, optional
        Number of dimensions in which grid is created, by default None
    format : str, optional
        'grid' -> returns meshgrid of size (Ndim, *grid_size)
        'vec' -> returns matrix of shape (Nsamples, Ndim) as vector
                of grid location for all grid points
        by default 'grid'

    Returns
    -------
    np.meshgrid : if format=='grid'
    np.ndarray : if format='vec'
    """
    grid_size, ndim = _checkdim(grid_size, ndim, "Gridsize")
    fov, ndim = _checkdim(fov, ndim, "FOV")

    dk = 1 / fov
    kmax = dk * grid_size / 2

    grid_ticks = []
    for g, k, d in zip(grid_size, kmax, dk):
        ticks = np.linspace(-k, k, g, endpoint=False) + d / 2
        grid_ticks.append(ticks)

    mesh_locs = np.meshgrid(*grid_ticks)

    if format == "grid":
        return mesh_locs
    elif format == "vec":
        return np.column_stack([ml.ravel() for ml in mesh_locs])


def decay(
    t: Union[float, np.ndarray],
    params: pd.DataFrame,
    flip: float,
    tr: float,
    t2star: bool = True,
) -> np.ndarray:
    """Calculate decay of steady state spin density for timepoints t.

    Parameters
    ----------
    t : float, np.ndarray
        Timepoints of the decay [s].
    params : pd.DataFrame
        Tissue/sequence parameters.
        keys: [spin, t1, t2 (or t2s)]
    flip : float
        Excitation flip angle [degree].
    tr : float
        Repetition time TR [s].
    t2star : bool
        Sequence is gradient echo sequence. Use T2* instead T2.

    Returns
    -------
    np.ndarray
        Decay of spin density for every timepoint t.

    Notes
    -----
    Assuming that the decay is happening in the whole field of view
    and the region of interest (eg. ellipsoid in Shepp Logan) is
    "binary" (1 in the region of interest and 0 outside)
    the decay can be viewed as spatial constant for the whole FOV and
    therefore can be multiplied to the FT of the geometry in kspace.

    References
    ----------
    ..[1] Gach, H. Michael, Costin Tanase, und Fernando Boada.
    „2D & 3D Shepp-Logan Phantom Standards for MRI“.
    2008. IEEE. https://doi.org/10.1109/ICSEng.2008.15.

    ..[2] https://www.medphysics.wisc.edu/~block/bme530lectures/mr1.pdf
    """
    flip = np.deg2rad(flip)

    transv = params.t2s
    if t2star is False:
        transv = params.t2

    decay = (
        np.sin(flip)
        * (1 - np.exp(-tr / params.t1))
        / (1 - np.cos(flip) * np.exp(-tr / params.t1))
        * np.exp(-t / transv)
    )
    return params.spin * decay


def find_quaternion(v1, v2) -> np.ndarray:
    """Calculate the quaternion which rotate vector v1 onto vector v2"""
    # Calculate the cross product of v1 and v2
    a = np.cross(v1, v2)

    # Calculate the components of the quaternion q
    q = np.zeros(4)  # Initialize the quaternion as [0, 0, 0, 0]

    # Set the xyz components of q to the calculated cross product (a)
    q[0:3] = a

    # Calculate the scalar component (w) of the quaternion
    q[3] = np.sqrt(np.linalg.norm(v1) ** 2 * np.linalg.norm(v2) ** 2) + np.dot(v1, v2)

    return q
