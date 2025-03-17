from typing import Union

import numpy as np
import pandas as pd

from . import utils as ut

# fmt: off
# ruff: noqa: F501
Ellipsoids_3D = pd.DataFrame({
    'center_x':     [0.00   , 0.00  , 0.00      , 0.00      , -0.22 , 0.22  , 0.00      , 0.0       , -0.08     , 0.06      , 0.0       , 0.00      , 0.06      , 0.00  , 0.56      ], # noqa: E501
    'center_y':     [0.00   , 0.00  , -0.0184   , -0.0184   , 0.000 , 0.00  , 0.35      , 0.1       , -0.605    , -0.605    , -0.10     , -0.605    , -0.105    , 0.1   , -0.4      ], # noqa: E501
    'center_z':     [0.00   , 0.00  , 0.00      , 0.0000    , -0.25 , -0.25 , -0.25     , -0.25     , -0.25     , -0.25     , -0.25     , -0.25     , 0.0625    , 0.625 , -0.25     ], # noqa: E501
    'axis_a':       [0.720  , 0.69  , 0.6624    , 0.6524    , 0.41  , 0.31  , 0.210     , 0.046     , 0.046     , 0.046     , 0.046     , 0.023     , 0.056     , 0.056 , 0.2       ], # noqa: E501
    'axis_b':       [0.95   , 0.92  , 0.874     , 0.864     , 0.16  , 0.11  , 0.25      , 0.046     , 0.023     , 0.023     , 0.046     , 0.023     , 0.04      , 0.056 , 0.03      ], # noqa: E501
    'axis_c':       [0.93   , 0.9   , 0.88      , 0.87      , 0.21  , 0.22  , 0.35      , 0.046     , 0.02      , 0.02      , 0.046     , 0.023     , 0.1       , 0.1   , 0.1       ], # noqa: E501
    'angle':        [0.0    , 0.0   , 0.0       , 0.0       , -72.0 , 72.0  , 0.0       , 0.0       , 0.0       , -90.0     , 0.0       , 0.0       , -90.0     , 0.0   , 70.0      ], # noqa: E501
    'tissue':       ['scalp', 'bone','csf'      , 'gray'    , 'csf' , 'csf' , 'white'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   , 'csf' , 'clot'    ], # noqa: E501
})

Ellipsoids_2D = pd.DataFrame({
    'center_x':     [0.00   , 0.00  , 0.00      , 0.00      , -0.22 , 0.22  , 0.00      , 0.0       , -0.08     , 0.06      , 0.0       , 0.00      ], # noqa: E501
    'center_y':     [0.00   , 0.00  , -0.0184   , -0.0184   , 0.000 , 0.00  , 0.35      , 0.1       , -0.605    , -0.605    , -0.10     , -0.605    ], # noqa: E501
    'axis_a':       [0.72   , 0.69  , 0.6624    , 0.6524    , 0.41  , 0.31  , 0.21      , 0.046     , 0.046     , 0.046     , 0.046     , 0.023     ], # noqa: E501
    'axis_b':       [0.95   , 0.92  , 0.874     , 0.864     , 0.16  , 0.11  , 0.25      , 0.046     , 0.023     , 0.023     , 0.046     , 0.023     ], # noqa: E501
    'angle':        [0.0    , 0.0   , 0.0       , 0.0       , -72.0 , 72.0  , 0.0       , 0.0       , 0.0       , -90.0     , 0.0       , 0.0       ], # noqa: E501
    'tissue':       ['scalp', 'bone','csf'      , 'gray'    , 'csf' , 'csf' , 'white'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   ], # noqa: E501
})

Tissue_params = pd.DataFrame({
    'spin'  : [0.8      , 0.12      , 0.98  , 0.85  , 0.745 , 0.617 , 0.95      ],
    't1fitA': [0.324    , 0.533     , None  , 1.35  , 0.857 , 0.583 , 0.926     ],
    't1fitC': [0.137    , 0.088     , None  , 0.34  , 0.376 , 0.382 , 0.217     ],
    't2'    : [0.07     , 0.05      , 1.99  , 0.2   , 0.1   , 0.08  , 0.1       ],
    'chi'   : [-7.5e-6  , -8.85e-6  , -9e-6 , -9e-6 , -9e-6 , -9e-6 , -9e-6     ],
    #'t2'    : [0.60     , 0.60      , 0.60  , 0.60   , 0.60   , 0.60  , 0.60       ],
    #'chi'   : [0  , 0  , 0 , 0 , 0 , 0 , 0     ],
    'tissue': ['scalp'  ,'bone'     ,'csf'  ,'clot' ,'gray' ,'white', 'tumor'   ]
})
# fmt: on


def getParams(
    fov: Union[float, np.ndarray, list],
    ndim=None,
    b0: float = 3.0,
    gamma: float = 42.576e6,
    blood_clot: bool = False,
    homogeneous: bool = True,
) -> pd.DataFrame:
    """Return MRI SheppLogan phantom params for given field strength
    and gyromag. ratio (of H).

    Parameters
    -----------
    fov: float or ndarray or list
        Field of view [m] in each dim or isotropic fov.
        If fov is a scalar, the parameter ndim must be set.
    ndim : int, optional
        Dimension of Phantom (2D or 3D). Defaults to None.
        If None the shape of fov defines the dimensions.
    b0 : float, optional
        Main magnetic field strength. Defaults to 3.0.
    gamma : float, optional
        Gyromagnetic ratio of H. Defaults to 42.576e6.
    Nellips : int, optional
        Number of ellipses parameters returned. Defaults to 15 (3D)
    homogeneous : bool, optional
        If homogeneous is True also parameters for ellipsoids
        with negative spin density will be returned for subtraction.
        Defaults to True.
    blood_clot : bool, optional
        Defines a blood clot at the side of the 3D Shep Logan Phantom

    Returns
    --------
    pandas.DataFrame
        Dataframe of params for each ellipsoid in phantom.

    References
    -----------
     ..[1]    Gach, H.M., Tanase, C., Boada, F.
            "2D & 3D Shepp-Logan Phantom Standards for MRI".
            2008. 19th International Conference on Systems Engineering.
            https://doi.org/10.1109/ICSEng.2008.15
    """
    fov, ndim = ut._checkdim(fov, ndim, "fov")

    # select phantom params needed
    pos_output = Ellipsoids_2D.copy() if ndim == 2 else Ellipsoids_3D.copy()
    pos_output = pos_output.merge(Tissue_params, on="tissue", how="left")

    # Scale the FOV of standard phantom to user input
    pos_output = _fovScale(fov, pos_output)

    # Calculate T1 and T2 values from fit like described in Gach et al.
    pos_output["t1"] = _T1(pos_output["t1fitA"].values, pos_output["t1fitC"].values, b0)

    pos_output["t2s"] = _T2s(
        pos_output["t2"].values, pos_output["chi"].values, b0, gamma
    )

    # Add a blood clot to the phantom... or don't
    if not blood_clot:
        pos_output = pos_output.drop(pos_output[pos_output["tissue"] == "clot"].index)

    # Dont add negative ellipsoids if not homogeneous
    if not homogeneous:
        return pos_output

    # Create ellipsoids with negative signal to get homogeneous phantom
    elif homogeneous:
        # copy positive params to make substraction starting from second ellipsoid
        neg_output = pos_output.copy().iloc[1:]

        for i in range(1, pos_output.shape[0]):
            if ndim == 2:
                geometrie = pos_output[
                    ["center_x", "center_y", "axis_a", "axis_b", "angle"]
                ].iloc[i]
            elif ndim == 3:
                geometrie = pos_output[
                    [
                        "center_x",
                        "center_y",
                        "center_z",
                        "axis_a",
                        "axis_b",
                        "axis_c",
                        "angle",
                    ]
                ].iloc[i]

            if i < 4:
                prop = pos_output[
                    ["spin", "tissue", "t1fitA", "t1fitC", "chi", "t2", "t1", "t2s"]
                ].iloc[i - 1]
            else:
                prop = pos_output[
                    ["spin", "tissue", "t1fitA", "t1fitC", "chi", "t2", "t1", "t2s"]
                ].iloc[3]

            new_row = pd.concat([geometrie, prop])

            neg_output.loc[i] = new_row

        neg_output["spin"] *= -1

        output = pd.concat([pos_output, neg_output], ignore_index=True)

        return output
    else:
        ValueError(
            f"Homogeneous parameter {homogeneous} invalid. Should be True or False."
        )


def _fovScale(fov: np.ndarray, loc_params: pd.DataFrame) -> pd.DataFrame:
    """Scales the location params of the ellipsoids to a given fov array [x, y, (,z)]
    fov has to be an array of shape (ndim,) [m."""
    ndim = len(fov)

    if ndim not in [2, 3]:
        ValueError(f"Number of dimensions is {ndim}. Allowed are ndim=2, ndim=3.")

    if ndim >= 2:
        loc_params.loc[:, ["center_x", "axis_a"]] *= fov[0] / 2
        loc_params.loc[:, ["center_y", "axis_b"]] *= fov[1] / 2

    if ndim == 3:
        loc_params.loc[:, ["center_z", "axis_c"]] *= fov[2] / 2

    return loc_params


def _T1(
    fitA: Union[float, np.ndarray], fitC: Union[float, np.ndarray], b0: float
) -> Union[float, np.ndarray]:
    """Calculate T1 values for tissue specific fit parameters.

    Parameters
    ----------
    fitA : float, np.ndarray
        Fit parameters A.
    fitC : float, np.ndarray
        Fit parameters C.
    b0 : float
        Main magnetic field strength in T.

    Returns
    -------
    float, np.ndarray
        T1 values for given fit parameters.

    Notes
    -----
    Implementation as described in [1].

    Returns T1 = A * (B0)^C

    For NaN values the function returns 4.2s as T1.

    References
    -----------
    ..[1]    Gach, H.M., Tanase, C., Boada, F.
            "2D & 3D Shepp-Logan Phantom Standards for MRI".
            2008. 19th International Conference on Systems Engineering.
            https://doi.org/10.1109/ICSEng.2008.15
    """
    # convert single float values to arrays
    if isinstance(fitA, float):
        fitA = np.array([fitA])
    if isinstance(fitC, float):
        fitC = np.array([fitC])

    T1 = fitA * b0**fitC

    # replace nan values
    T1 = np.where(np.isnan(T1), 4.2, T1)

    return T1


def _T2s(
    t2: Union[float, np.ndarray], chi: Union[float, np.ndarray], b0: float, gamma: float
) -> Union[float, np.ndarray]:
    """Calculate T2* values.

    Parameters
    ----------
    t2 : Union[float, np.ndarray].ArrayLike
        T2 times [s].
    chi : Union[float, np.ndarray].ArrayLike
        Magnetic susceptibility.
    b0 : float
        Main magnetic field strength [T].
    gamma : float
        Gyromagnetic ration [Hz/T].

    Returns
    -------
    ts2 : Union[float, np.ndarray].ArrayLike
        T2* times.

    Notes
    -----

    Returns T2* = 1/T2 + gamma/2pi * |b0 * chi|

    References
    ----------
    ..[1]    Gach, H.M., Tanase, C., Boada, F.
            "2D & 3D Shepp-Logan Phantom Standards for MRI".
            2008. 19th International Conference on Systems Engineering.
            https://doi.org/10.1109/ICSEng.2008.15
    """
    # convert single float values to arrays
    if isinstance(t2, float):
        t2 = np.array([t2])
    if isinstance(chi, float):
        chi = np.array([chi])

    r2s = 1 / t2 + gamma / (2 * np.pi) * np.abs(b0 * chi * 1e-6)

    t2s = 1 / r2s

    return t2s
