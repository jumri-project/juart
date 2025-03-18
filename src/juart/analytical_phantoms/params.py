from typing import Literal, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from . import utils as ut


def signal_equation(seq_type: str, **kwargs):
    """Calculate the signal for a given sequence type and sequence parameters.

    Parameters
    ----------
    seq_type : str
        Sequence type (e.g. 'GRE', {'SE', 'IR', 'SSFP' are not supported}).
    **kwargs:
        Sequence parameters:
        - 'spin_density': float
            Spin density of the tissue [1/m^3].
        - 't1': float
            Longitudinal relaxation time [s].
        - 't2p': float
            Transverse relaxation time [s].
        - 'tr': float
            Repetition time [s].
        - 'te': float or
            Echo time [s].
        - 'flip': float
            Flip angle [rad].

    Returns
    -------
    Singal : np.ndarray
        Signal for the given sequence type and parameters for every `te` in **kwargs.

    """
    if isinstance(kwargs["te"], float):
        kwargs["te"] = [kwargs["te"]]

    kwargs["te"] = np.asarray(kwargs["te"])

    if seq_type == "GRE":
        signal = kwargs["spin_density"] * (
            np.sin(kwargs["flip"])
            * (1 - np.exp(-kwargs["tr"] / kwargs["t1"]))
            / (1 - np.cos(kwargs["flip"]) * np.exp(-kwargs["tr"] / kwargs["t1"]))
            * np.exp(-kwargs["te"] / kwargs["t2p"])
        )

        return signal


class Geometry:
    """Class for Ellipsoid with position parameters."""

    def __init__(
        self,
        center: npt.ArrayLike,
        axes: npt.ArrayLike,
        angle: float,
        device: str = "cpu",
    ):
        self.device = device
        self.center = torch.as_tensor(center, dtype=torch.float32, device=self.device)
        self.axes = torch.as_tensor(axes, dtype=torch.float32, device=self.device)
        self.angle = torch.as_tensor(
            angle, dtype=torch.float32, device=self.device
        )  # in rad
        # Ellipsoid has to be 2D or 3D
        if self.center.shape[0] not in [2, 3]:
            raise ValueError("Center has to be an array of shape (2,) or (3,).")
        if self.axes.shape[0] not in [2, 3]:
            raise ValueError("Axes has to be an array of shape (3,).")

    @property
    def ndim(self):
        return self.center.shape[0]

    @property
    def rot_matrix(self):
        if self.ndim == 2:
            rot_mat = torch.tensor(
                [
                    [torch.cos(self.angle), -torch.sin(self.angle)],
                    [torch.sin(self.angle), torch.cos(self.angle)],
                ],
                device=self.device,
                dtype=torch.float32,
            )
        elif self.ndim == 3:
            rot_mat = torch.tensor(
                [
                    [torch.cos(self.angle), -torch.sin(self.angle), 0],
                    [torch.sin(self.angle), torch.cos(self.angle), 0],
                    [0, 0, 1],
                ],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                "Rotation matrix is only defined for 2D and 3D ellipsoids."
            )

        return rot_mat

    def scale(self, scale: Union[float, torch.tensor]):
        """Scale the ellipsoid by a scalar factor
        or by a factor for each dimension (xyz)."""
        if isinstance(scale, (int, float)):
            self.center *= scale
            self.axes *= scale
        else:
            scale = torch.as_tensor(scale, device=self.device)
            if scale.shape[0] != 3:
                raise ValueError("Scale has to be a scalar or an array of shape (3,).")
            self.center *= scale
            self.axes *= scale

    def get_support(self, r: torch.tensor) -> torch.tensor:
        """Returns a mask which elements of the input r are inside the ellipsoid.

        Parameters
        ----------
        r : tensor, Shape (D, N)
            Location samples for which support should be returned [m].

        Returns
        -------
        tensor, Shape (N,)
            Support mask which samples of "r" are inside the ellipsoid.

        Raises
        ------
        ValueError
            If r has not the same number of dimensions as the ellipsoid.
        """
        r = torch.as_tensor(r, device=self.device, dtype=torch.float32)
        num_dim, num_samples = r.shape

        if num_dim != self.ndim:
            raise ValueError(
                "r has to have the same number of dimensions as the ellipsoid."
            )

        # Transform ellipsoid back to unit sphere
        ellip_shift = r - self.center[:, None] / 2

        ellip_rot = torch.matmul(self.rot_matrix, ellip_shift)

        sphere = ellip_rot**2 / self.axes[:, None] ** 2

        support = sphere.sum(dim=0) <= 1

        return support


class Tissue:
    """Class for tissue parameters."""

    def __init__(
        self,
        spin_density: float,
        t1_fitA: float,
        t1_fitC: float,
        t2: float,
        chi: float,
        type: str,
    ):
        self.spin_density = spin_density
        self.t1_fitA = t1_fitA
        self.t1_fitC = t1_fitC
        self.t2 = t2  # Assume T2 does not change with B0
        self.chi = chi
        self.type = type

    def get_t1(self, b0: float):
        """Calculates the T1 relaxation time in seconds
        based on the main magnetic field strength (B0).

        Parameters
        ----------
        b0 : float
            The main magnetic field strength in Tesla [T].

        Returns
        -------
        t1:
            The calculated T1 relaxation time in seconds.
            If the parameters `t1_fitA` or `t1_fitC` are not set,
            a default value of 4.2 seconds is returned.
            Otherwise, the T1 time is computed using the formula:
            T1 = t1_fitA * B0^t1_fitC.
        """
        if self.t1_fitA is None or self.t1_fitC is None:
            return 4.2
        else:
            return self.t1_fitA * b0**self.t1_fitC

    def get_t2s(self, b0: float, gamma: float):
        """Calculate T2* values.

        Parameters
        ----------
        b0 : float
            Main magnetic field in [T]
        gamma : float
            Gyromag. ratio in [Hz/T]

        Returns
        -------
        t2s: float
            T2* relaxation time in [s]
        """
        return 1 / (1 / self.t2 + gamma / (2 * np.pi) * np.abs(b0 * self.chi * 1e-6))

    def get_signal(self, seq_type: str = "GRE", **kwargs):
        """Get the signal
        of the tissue for a given sequence type and sequence parameters.

        Parameters
        ----------
        seq_type : str, optional
            Sequence type for which to calculate the signal, by default 'GRE'
        **kwargs:
            Sequence parameters:
            - 'te': float
                Echo time [s].
            - 'tr': float
                Repetition time [s].
            - 'flip': float
                Flip angle [rad].
            - 'b0': float
                Main magnetic field strength [T].
            - 'gamma': float
                Gyromagnetic ratio of H [Hz/T].

        Returns
        -------
        signal : np.ndarray, Shape (E,)
            Signal of the tissue for the given sequence type and parameters.
            If `te` is a scalar, E = 1.
            If `te` is an array of size E, the signal is an array of size E.

        """
        if kwargs is None:
            kwargs = {}

        # Default values
        defaults = {
            "te": 5e-3,
            "tr": 2,
            "flip": np.deg2rad(30),
            "b0": 3.0,
            "gamma": 42.576e6,
        }

        # Update kwargs with defaults if not already defined
        for key, value in defaults.items():
            kwargs.setdefault(key, value)

        t1 = self.get_t1(kwargs["b0"])

        if seq_type == "GRE":
            t2p = self.get_t2s(kwargs["b0"], kwargs["gamma"])
        else:
            t2p = self.t2

        kwargs["t1"] = t1
        kwargs["t2p"] = t2p
        kwargs["spin_density"] = self.spin_density

        signal = signal_equation(seq_type=seq_type, **kwargs)

        return signal


class Ellipsoid:
    """Class for Ellipsoid with tissue parameters."""

    def __init__(
        self,
        geometry: Geometry,
        tissue: Tissue,
        counter: int,
    ):
        self.geometry = geometry
        self.tissue = tissue
        self.counter = counter

    @property
    def device(self):
        return self.geometry.device

    @property
    def ndim(self):
        return self.geometry.ndim

    def get_object(
        self, matrix: torch.tensor, fov: torch.tensor, seq_type: str, seq_params: dict
    ):
        """Generate the signal object for the ellipsoid.

        Parameters
        ----------
        matrix : torch.tensor, Shape (D,)
            Number of grid points along each dimension.
        fov : torch.tensor, Shape (D,)
            Field of view in meters for each dimension.
        seq_type : str, optional
            Sequence type for which to calculate the signal, by default 'GRE'.
        seq_params : Optional[dict], optional
            Sequence parameters
            such as 'te', 'tr', 'flip', 'b0', and 'gamma', by default None.
        device : str, optional
            Device on which to calculate the signal, by default 'cpu'.
        Returns
        -------
        np.ndarray, Shape (M, *grid_size, E)
            Signal object for the ellipsoid. The shape depends on the grid size and the
            number of echo times `te` (E) contained in `seq_params`.
        """
        matrix = torch.asarray(matrix, dtype=torch.int32, device=self.device)
        fov = torch.asarray(fov, dtype=torch.float32, device=self.device)

        if not matrix.shape[0] == fov.shape[0]:
            raise ValueError("Matrix and fov have to have the same length.")
        if not matrix.shape[0] == self.ndim:
            raise ValueError(
                "Matrix and ellipsoid have to have the same number of dimensions."
            )

        loc = create_image_grid_locations(
            matrix, fov, format="grid", device=self.device
        )
        num_dim, *grid_shape = loc.shape

        # Reshape to (ndim, Nsamples)
        loc = loc.view(self.ndim, -1)

        support = self.geometry.get_support(loc)

        signal = torch.as_tensor(
            self.tissue.get_signal(seq_type=seq_type, **seq_params),
            dtype=torch.float32,
            device=self.device,
        )
        num_echoes = signal.shape[0]

        signal_obj = support[..., None] * signal

        return signal_obj.view((*grid_shape, num_echoes))


def create_image_grid_locations(
    grid_size: torch.tensor,
    fov: torch.tensor,
    format: Literal["grid", "vec"] = "vec",
    device: str = "cpu",
):
    """Creates a cartesian grid in image space from [-fov/2, +fov/2] in real
    si units locations r [m].

    Parameters
    ----------
    grid_size : Tensor, Shape (M,)
        Grid size of the cartesian grid.
    fov : Tensor, Shape (M,)
        Field of view.
    format : str, optional
        'grid' -> returns meshgrid of size (Ndim, *grid_size)
        'vec' -> returns matrix of shape (Nsamples, Ndim) as vector
                of grid location for all grid points
        by default 'grid'
    device : str, optional
        Device on which to calculate the signal, by default 'cpu'.

    Returns
    -------
    ndarrray, Shape (M, *grid_size) : if format=='grid'
    ndarray, Shape (M, prod(grid_size)) : if format='vec'
    """

    grid_size = torch.as_tensor(grid_size, dtype=torch.int32, device=device)
    fov = torch.as_tensor(fov, dtype=torch.float32, device=device)

    if not (grid_size.size() == fov.size()):
        raise ValueError("grid_size and fov have to have the same length.")

    # Create a grid in image space which contains the
    # locations of the pixel centres of the grid
    grid_ticks = []

    for g, f in zip(grid_size, fov):
        dr = f / g
        gt = torch.arange(g, device=fov.device, dtype=fov.dtype) * dr - f / 2 + dr / 2
        grid_ticks.append(gt)

    mesh_locs = torch.meshgrid(*grid_ticks, indexing="ij")
    mesh_locs = torch.stack(mesh_locs, dim=0)

    if format == "grid":
        return mesh_locs

    elif format == "vec":
        return mesh_locs.view(mesh_locs.shape[0], -1)

    else:
        raise ValueError("format has to be 'grid' or 'vec'.")


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
