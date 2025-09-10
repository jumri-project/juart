import math
import warnings
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from .coils import Coil


def signal_equation(**kwargs):
    """Calculate the signal for a given sequence type and sequence parameters.

    Parameters
    ----------
    **kwargs:
        Sequence parameters:
        - 'seq_type': str
            Sequence type for which to calculate the signal.
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

    if kwargs["seq_type"] == "GRE":
        signal = kwargs["spin_density"] * (
            np.sin(kwargs["flip"])
            * (1 - np.exp(-kwargs["tr"] / kwargs["t1"]))
            / (1 - np.cos(kwargs["flip"]) * np.exp(-kwargs["tr"] / kwargs["t1"]))
            * np.exp(-kwargs["te"] / kwargs["t2p"])
        )

        return signal


class Geometry:
    """Class for Ellipsoid geometry."""

    def __init__(
        self,
        center: npt.ArrayLike,
        axes: npt.ArrayLike,
        angle: float,
        device: Optional[str] = None,
    ):
        self.center = torch.as_tensor(center, dtype=torch.float32, device=device)
        self.axes = torch.as_tensor(axes, dtype=torch.float32, device=device)
        self.angle = torch.as_tensor(
            angle, dtype=torch.float32, device=device
        )  # in rad
        # Ellipsoid has to be 2D or 3D
        if self.center.shape[0] not in [2, 3]:
            raise ValueError("Center has to be an array of shape (2,) or (3,).")
        if self.axes.shape[0] not in [2, 3]:
            raise ValueError("Axes has to be an array of shape (3,).")

    @classmethod
    def from_dict(
        cls,
        d: dict,
        device: Optional[str] = None,
    ):
        """Create a Geometry object from a dictionary."""
        if "center_z" in d:
            center = [d["center_x"], d["center_y"], d["center_z"]]
            axes = [d["axis_a"], d["axis_b"], d["axis_c"]]
            angle = math.radians(d["angle"])
        else:
            center = [d["center_x"], d["center_y"]]
            axes = [d["axis_a"], d["axis_b"]]
            angle = math.radians(d["angle"])

        return cls(center, axes, angle, device)

    @property
    def device(self):
        return self.center.device

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
            self.center *= scale / 2
            self.axes *= scale / 2
        else:
            scale = torch.as_tensor(scale, device=self.device)
            if scale.shape[0] != self.ndim:
                raise ValueError("Scale has to be a scalar or an array of shape (3,).")
            self.center *= scale / 2
            self.axes *= scale / 2

    def get_obj_vec(self, r: torch.tensor) -> torch.tensor:
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
        ellip_shift = r - self.center[:, None]

        ellip_rot = torch.matmul(self.rot_matrix.T, ellip_shift)

        sphere = ellip_rot**2 / self.axes[:, None] ** 2

        support = sphere.sum(dim=0) <= 0.98  # Adjust for small rounding errors

        return support

    def get_obj_grid(self, matrix: torch.tensor, fov: torch.tensor):
        """Generate the support mask for the ellipsoid on a grid with size `matrix`.

        Parameters
        ----------
        matrix : torch.tensor, Shape (D,)
            Number of grid points along each dimension.
        fov : torch.tensor, Shape (D,)
            Field of view in meters for each dimension.

        Returns
        -------
        support : torch.tensor, Shape (*grid_size)
            Support mask for the ellipsoid on the grid.
        """
        matrix = torch.as_tensor(matrix, dtype=torch.int32, device=self.device)
        fov = torch.as_tensor(fov, dtype=torch.float32, device=self.device)

        loc = create_image_grid_locations(
            matrix, fov, format="grid", device=self.device
        )

        num_dim, *grid_shape = loc.shape

        loc = loc.view(self.ndim, -1)

        support = self.get_obj_vec(loc)

        return support.view(grid_shape)

    def get_ksp_vec(self, ktraj: torch.tensor) -> torch.tensor:
        """Calculate the k-space signal for the ellipsoid geometry.

        Parameters
        ----------
        ktraj : torch.tensor, Shape (D, N)
            Kspace sample locations in D dimensions with N samples
            for which the signal should be calculated [1/m].

        Returns
        -------
        coeff, torch.tensor (N,)
            Kspace coefficients of the ellipsoid's geometry
            for the given k-space samples `ktraj`.

        Raises
        ------
        ValueError
            If D is not equal to the number of dimensions of the ellipsoid.
        """
        num_dim, num_samples = ktraj.shape

        if ktraj.device != self.device:
            ktraj = ktraj.to(self.device)
            warnings.warn(
                f"'ktraj' is on device {ktraj.device} but class is on {self.device}."
                f"Copied `ktraj` to device {self.device}.",
                stacklevel=2,
            )

        if num_dim != self.ndim:
            raise ValueError(
                "ktraj has to have the same number of dimensions as the ellipsoid."
            )

        # Difference between 2D and 3D k-space
        if self.ndim == 2:
            exp = torch.exp(
                -1j
                * 2
                * math.pi
                * (ktraj[0] * self.center[0] + ktraj[1] * self.center[1])
            )
            # Save some computations
            rot_mat = self.rot_matrix

            kappa = torch.sqrt(
                (self.axes[0] * (ktraj[0] * rot_mat[0, 0] + ktraj[1] * rot_mat[1, 0]))
                ** 2
                + (self.axes[1] * (ktraj[0] * rot_mat[0, 1] + ktraj[1] * rot_mat[1, 1]))
                ** 2
            )

            # Calculate bessel function
            near_zero = kappa < 0.001

            k_bessel = torch.empty(kappa.shape[0], device=self.device)

            k_bessel[~near_zero] = torch.special.bessel_j1(
                2 * math.pi * kappa[~near_zero]
            ) / (math.pi * kappa[~near_zero])

            k_bessel[near_zero] = (
                1
                - 1 / 2 * (math.pi * kappa[near_zero]) ** 2
                + 1 / 12 * (math.pi * kappa[near_zero]) ** 4
            )

            coeff = math.pi * torch.prod(self.axes) * exp * k_bessel

            return coeff

        if self.ndim == 3:
            exp = torch.exp(
                -1j
                * 2
                * math.pi
                * (
                    ktraj[0] * self.center[0]
                    + ktraj[1] * self.center[1]
                    + ktraj[2] * self.center[2]
                )
            )

            rot_mat = self.rot_matrix

            kappa = torch.sqrt(
                (self.axes[0] * (ktraj[0] * rot_mat[0, 0] + ktraj[1] * rot_mat[1, 0]))
                ** 2
                + (self.axes[1] * (ktraj[0] * rot_mat[0, 1] + ktraj[1] * rot_mat[1, 1]))
                ** 2
                + (self.axes[2] * ktraj[2]) ** 2
            )

            near_zero = kappa < 0.001

            k_bessel = torch.empty(kappa.shape[0], device=self.device)

            k_bessel[~near_zero] = (
                torch.sin(2 * math.pi * kappa[~near_zero])
                - 2
                * math.pi
                * kappa[~near_zero]
                * torch.cos(2 * math.pi * kappa[~near_zero])
            ) / (2 * math.pi**2 * kappa[~near_zero] ** 3)

            k_bessel[near_zero] = (
                4 / 3 * math.pi
                - 8 / 15 * math.pi**3 * kappa[near_zero] ** 2
                + 8 / 105 * math.pi**5 * kappa[near_zero] ** 5
            )

            coeff = torch.prod(self.axes) * exp * k_bessel

            return coeff


class Tissue:
    """Class for tissue parameters."""

    def __init__(
        self,
        spin_density: float,
        t2: float,
        chi: float,
        type: str,
        t1_fitA: Optional[float] = None,
        t1_fitC: Optional[float] = None,
    ):
        self.spin_density = spin_density
        self.t1_fitA = t1_fitA
        self.t1_fitC = t1_fitC
        self.t2 = t2  # Assume T2 does not change with B0
        self.chi = chi
        self.type = type

    @classmethod
    def from_dict(cls, d: dict):
        """Create a Tissue object from a dictionary."""
        output = cls(
            spin_density=d["spin_density"],
            t1_fitA=None if np.isnan(d["t1_fitA"]) else d["t1_fitA"],
            t1_fitC=None if np.isnan(d["t1_fitC"]) else d["t1_fitC"],
            t2=d["t2"],
            chi=d["chi"],
            type=d["type"],
        )

        return output

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
        return 1 / (1 / self.t2 + gamma / (2 * math.pi) * np.abs(b0 * self.chi * 1e-6))

    def get_signal(self, **kwargs):
        """Get the signal
        of the tissue for a given sequence type and sequence parameters.

        Parameters
        ----------
        **kwargs:
            Sequence parameters:
            - 'seq_type': str
                Sequence type for which to calculate the signal.
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
        t1 = self.get_t1(kwargs["b0"])

        if kwargs["seq_type"] == "GRE":
            t2p = self.get_t2s(kwargs["b0"], kwargs["gamma"])
        else:
            t2p = self.t2

        kwargs["t1"] = t1
        kwargs["t2p"] = t2p
        kwargs["spin_density"] = self.spin_density

        signal = signal_equation(**kwargs)

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

    @classmethod
    def from_dict(
        cls,
        d: dict,
        counter: int,
        device: Optional[str] = None,
    ):
        """Create an Ellipsoid object from a dictionary."""
        geometry = Geometry.from_dict(d, device)
        tissue = Tissue.from_dict(d)
        return cls(geometry, tissue, counter)

    @property
    def device(self):
        return self.geometry.device

    @property
    def ndim(self):
        return self.geometry.ndim

    def scale(self, scale: Union[float, torch.tensor]):
        """Scale the ellipsoid by a scalar factor
        or by a factor for each dimension (xyz)."""
        self.geometry.scale(scale)

    def get_object(
        self,
        matrix: torch.tensor,
        fov: torch.tensor,
        seq_params: Optional[dict] = None,
    ):
        """Generate the signal object for the ellipsoid.

        Parameters
        ----------
        matrix : torch.tensor, Shape (D,)
            Number of grid points along each dimension.
        fov : torch.tensor, Shape (D,)
            Field of view in meters for each dimension.
        seq_type : str, optional
            Sequence type for which to calculate the signal, by default 'GRE'
        seq_params : dict, optional
            Sequence parameters, by default a default set of GRE parameters is used.

        Returns
        -------
        signal_obj : torch.tensor, Shape (*grid_size, E)
            Signal object for the ellipsoid on the grid with E echoes.
        """

        support = self.get_object_geometry(matrix, fov)

        if seq_params is None:
            signal = self.tissue.spin_density
        else:
            signal = torch.as_tensor(
                self.tissue.get_signal(**seq_params),
                dtype=torch.float32,
                device=self.device,
            )

        signal_obj = support[..., None] * signal

        return signal_obj

    def get_object_geometry(self, matrix: torch.tensor, fov: torch.tensor):
        """Generate the signal object for the ellipsoid.

        Parameters
        ----------
        matrix : torch.tensor, Shape (D,)
            Number of grid points along each dimension.
        fov : torch.tensor, Shape (D,)
            Field of view in meters for each dimension.
        Returns
        -------
        np.ndarray, Shape (*grid_size)
            Geometry of the ellipsoid on the grid with intensities 0 or 1.
        """
        return self.geometry.get_obj_grid(matrix, fov)

    def get_ksp_signal(
        self,
        ktraj: torch.tensor,
        seq_params: Optional[dict] = None,
    ) -> torch.tensor:
        """Calculate the k-space signal for the ellipsoid geometry.

        Parameters
        ----------
        ktraj : torch.tensor, Shape (D, N)
            Kspace sample locations in D dimensions with N samples
            for which the signal should be calculated [1/m].
        seq_params : dict, optional
            Sequence parameters, by default only the spin-density is used.

        Returns
        -------
        signal : torch.tensor, Shape (N, E)
            K-space signal for the given k-space samples `ktraj`
            at echo times `te` in `seq_params`. \n
            If `te` is a scalar, the signal is a vector of size (N, 1) and all
            samples in `ktraj` are assumed to be sampled at the same `te`. \n
            If `te` is an array of size E, the signal is a matrix of size (N, E) and
            each column corresponds to the signal at the corresponding `te`. \n
            If `te` is an array of size E and E=N, the signal is a vector of size
            (N, 1), and each element corresponds to the signal at the corresponding
            `te`.\
        """

        geom_sig = self.geometry.get_ksp_vec(ktraj)

        if seq_params is None:
            tissue_sig = torch.as_tensor(
                [self.tissue.spin_density],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            tissue_sig = torch.as_tensor(
                self.tissue.get_signal(**seq_params),
                dtype=torch.float32,
                device=self.device,
            )

        num_dim, num_samples = ktraj.shape

        if tissue_sig.size(0) != num_samples:
            output = geom_sig[:, None] * tissue_sig
        else:
            output = (geom_sig * tissue_sig)[:, None]

        return output

    def get_kspace_geometry(self, ktraj: torch.tensor) -> torch.tensor:
        """Calculate the k-space signal for the ellipsoid geometry.

        Parameters
        ----------
        ktraj : torch.tensor, Shape (D, N)
            Kspace sample locations in D dimensions with N samples
            for which the signal should be calculated [1/m].

        Returns
        -------
        coeff, torch.tensor (N,)
            Kspace coefficients of the ellipsoid's geometry
            for the given k-space samples `ktraj`.

        Raises
        ------
        ValueError
            If D is not equal to the number of dimensions of the ellipsoid.
        """
        return self.geometry.get_ksp_vec(ktraj)


class SheppLogan:
    def __init__(
        self,
        fov: torch.tensor,
        matrix: torch.tensor,
        ts2: bool = True,
        blood_clot: bool = False,
        homogeneous: bool = True,
        device: Optional[str] = None,
    ):
        self.fov = torch.as_tensor(fov, dtype=torch.float32, device=device)
        self.matrix = torch.as_tensor(matrix, dtype=torch.int32, device=device)

        self.ts2 = ts2
        self.blood_clot = blood_clot
        self.homogeneous = homogeneous
        self.coil = None

        # Create the ellipsoids of phantom
        self.ellipsoids = self._create_ellipsoids()

    @property
    def ndim(self):
        return self.fov.size(0)

    @property
    def device(self):
        return self.fov.device

    def add_coil(self, coil: Optional[Coil] = None):
        """Add a coil to the phantom.

        Parameters
        ----------
        coil: Coil, optional
            Coil object to add to the phantom. If None, a default coil is created.
            Default coil has 8 channels for 2D case and 15 channels for 3D case.
        """
        if coil is None:
            r = torch.max(self.fov) / 2 + 0.05
            if self.ndim == 2:
                z = 0
                phi0 = 0
                num_channels_ring = 8
            elif self.ndim == 3:
                z = [-self.fov[2] / 4, 0, +self.fov[2] / 4]
                num_channels_ring = 5
                phi0 = [0, 2 * math.pi / num_channels_ring, 0]

            coil = Coil(
                coil_radius=r,
                num_loops_ring=num_channels_ring,
                z_pos=z,
                phi0=phi0,
            )

            # Adjust coil shape to ellipsoid shape of phantom
            a, b = self.ellipsoids[0].geometry.axes[:2]
            for i in range(len(coil.coil_loops)):
                phi = np.arctan2(
                    coil.coil_loops[i].r_cent[1], coil.coil_loops[i].r_cent[0]
                )

                new_r = 3 / 2 * self.fov[0] + np.sqrt(
                    a * np.cos(phi) ** 2 + b * np.sin(phi) ** 2
                )

                coil.coil_loops[i].r_cent = np.array(
                    [
                        new_r * np.cos(phi),
                        new_r * np.sin(phi),
                        coil.coil_loops[i].r_cent[2],
                    ]
                )

                coil.coil_loops[i]._build_coil_elements()

            self.coil = coil

        else:
            self.coil = coil

    def get_arb_kspace(
        self,
        ktraj: torch.Tensor,
        seq_params: Optional[dict] = None,
        type: Literal["analytic", "numeric"] = "numeric",
    ) -> torch.Tensor:
        """Returns the k-space signal for arbitrary kspace locations `ktraj`.

        Parameters
        ----------
        ktraj : torch.tensor, Shape (D, N)
            Kspace sample locations in D dimensions with N samples
            for which the signal should be calculated [1/m].
        seq_params : dict, optional
            Sequence parameters, by default each ellipsoids signal
            is its spin density.

        Returns
        -------
        signal_obj : torch.tensor, Shape (N, E)
            K-space signal for the ellipsoid on the grid with E echoes.
            E is definded by the number of echo times `te` in `seq_params`.
            If `seq_params` is None, E=1.
        """
        num_echoes = 1 if seq_params is None else len(seq_params["te"])

        signal_obj = torch.zeros(
            ktraj.shape[1], num_echoes, dtype=torch.complex64, device=self.device
        )

        for ellipsoid in self.ellipsoids:
            sig_ellipsoid = ellipsoid.get_ksp_signal(ktraj, seq_params)

            signal_obj += sig_ellipsoid

        return signal_obj

    def get_object(self, seq_params: Optional[dict] = None) -> torch.Tensor:
        """Generate the signal object for the ellipsoid.

        Returns
        -------
        signal_obj : torch.tensor, Shape (*grid_size, E)
            Signal object for the ellipsoid on the grid with E echoes.
            E is defined by the number of echo times `te` in `seq_params`.
            If `seq_params` is None, E=1.
        seq_params: dict, optional
            Sequence parameters, by default each ellipsoids signal
            is its spin density.
        """
        num_echoes = 1 if seq_params is None else len(seq_params["te"])

        signal_obj = torch.zeros(
            *self.matrix, num_echoes, dtype=torch.float32, device=self.device
        )

        for ellipsoid in self.ellipsoids:
            sig_ellipsoid = ellipsoid.get_object(self.matrix, self.fov, seq_params)

            signal_obj += sig_ellipsoid

        # TODO: All outputs should have shape (C, Nx, Ny, Nz, ...)

        if self.coil is not None:
            sens_maps = self.coil.get_sens_maps(self.matrix, self.fov)
            # Sens maps has shape (C, Nx, Ny, Nz) and has
            # to be adjusted to signal obj with shape (*matrix, E)
            # Add missing matrix dim
            if self.ndim == 2:
                signal_obj = signal_obj[:, :, None, ...]
            # Add missing channel dim
            signal_obj = signal_obj[None, ...]

            signal_obj *= sens_maps[..., None]  # Add echo dim to sensitivity maps

        return signal_obj

    def _create_ellipsoids(self) -> list[Ellipsoid]:
        ellipsoids = []
        ellips_counter = 0

        if self.ndim == 2:
            ellips_params = Ellipsoids_2D.copy()
        else:
            ellips_params = Ellipsoids_3D.copy()

        tissue_params = Tissue_params.copy()

        for n_ellipse in range(ellips_params.shape[0]):
            # Load the parameters for the current ellipsoid
            geom_params = ellips_params.loc[n_ellipse].to_dict()
            tissue_type = geom_params["type"]
            tissue_params = Tissue_params.loc[
                Tissue_params["type"].to_list().index(tissue_type)
            ].to_dict()

            # Do not add blood clot if not specified
            if tissue_type == "clot" and not self.blood_clot:
                continue

            d = {**geom_params, **tissue_params}

            ellipsoid = Ellipsoid.from_dict(d, ellips_counter)
            ellipsoid.scale(self.fov)

            ellipsoids.append(ellipsoid)

            ellips_counter += 1

        if not self.homogeneous:
            return ellipsoids
        else:
            # Create negativ signal ellipsoids
            # to subtract signal from overlapping ellipsoids
            # Yes this is not the most beautiful solution
            # but it works and is easy to implement
            for n_ellipse in range(1, ellips_params.shape[0]):
                geom_params = ellips_params.loc[n_ellipse].to_dict()

                if geom_params["type"] == "clot":
                    continue

                if n_ellipse < 4:
                    neg_params = ellips_params.loc[n_ellipse - 1].to_dict()
                else:
                    neg_params = ellips_params.loc[3].to_dict()

                tissue_type = neg_params["type"]
                tissue_params = Tissue_params.loc[
                    Tissue_params["type"].to_list().index(tissue_type)
                ].to_dict()

                # Do not add blood clot if not specified
                if tissue_type == "clot" and not self.blood_clot:
                    continue

                d = {**geom_params, **tissue_params}

                ellipsoid = Ellipsoid.from_dict(d, ellips_counter)
                ellipsoid.tissue.spin_density *= -1
                ellipsoid.scale(self.fov)

                ellipsoids.append(ellipsoid)
                ellips_counter += 1

            return ellipsoids


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
    'center_x':     [0.00   , 0.00  , 0.00      , 0.00      , -0.22 , 0.22  , 0.00      , 0.0       , -0.08     , 0.06      , 0.0       , 0.00      , 0.06      , 0.00   , 0.56      ], # noqa: E501
    'center_y':     [0.00   , 0.00  , -0.0184   , -0.0184   , 0.000 , 0.00  , 0.35      , 0.1       , -0.605    , -0.605    , -0.10     , -0.605    , -0.105    , 0.1    , -0.4      ], # noqa: E501
    'center_z':     [0.00   , 0.00  , 0.00      , 0.0000    , -0.25 , -0.25 , -0.25     , -0.25     , -0.25     , -0.25     , -0.25     , -0.25     , 0.0625    , 0.625  , -0.25     ], # noqa: E501
    'axis_a':       [0.720  , 0.69  , 0.6624    , 0.6524    , 0.41  , 0.31  , 0.210     , 0.046     , 0.046     , 0.046     , 0.046     , 0.023     , 0.056     , 0.056  , 0.2       ], # noqa: E501
    'axis_b':       [0.95   , 0.92  , 0.874     , 0.864     , 0.16  , 0.11  , 0.25      , 0.046     , 0.023     , 0.023     , 0.046     , 0.023     , 0.04      , 0.056  , 0.03      ], # noqa: E501
    'axis_c':       [0.93   , 0.9   , 0.88      , 0.87      , 0.21  , 0.22  , 0.35      , 0.046     , 0.02      , 0.02      , 0.046     , 0.023     , 0.1       , 0.1    , 0.1       ], # noqa: E501
    'angle':        [0.0    , 0.0   , 0.0       , 0.0       , -72.0 , 72.0  , 0.0       , 0.0       , 0.0       , -90.0     , 0.0       , 0.0       , -90.0     , 0.0    , 70.0      ], # noqa: E501
    'type':         ['scalp', 'bone','csf'      , 'gray'    , 'csf' , 'csf' , 'white'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   , 'csf'  , 'clot'    ], # noqa: E501
})

Ellipsoids_2D = pd.DataFrame({
    'center_x':     [0.00   , 0.00  , 0.00      , 0.00      , -0.22 , 0.22  , 0.00      , 0.0       , -0.08     , 0.06      , 0.0       , 0.00      ], # noqa: E501
    'center_y':     [0.00   , 0.00  , -0.0184   , -0.0184   , 0.000 , 0.00  , 0.35      , 0.1       , -0.605    , -0.605    , -0.10     , -0.605    ], # noqa: E501
    'axis_a':       [0.72   , 0.69  , 0.6624    , 0.6524    , 0.41  , 0.31  , 0.21      , 0.046     , 0.046     , 0.046     , 0.046     , 0.023     ], # noqa: E501
    'axis_b':       [0.95   , 0.92  , 0.874     , 0.864     , 0.16  , 0.11  , 0.25      , 0.046     , 0.023     , 0.023     , 0.046     , 0.023     ], # noqa: E501
    'angle':        [0.0    , 0.0   , 0.0       , 0.0       , -72.0 , 72.0  , 0.0       , 0.0       , 0.0       , -90.0     , 0.0       , 0.0       ], # noqa: E501
    'type':       ['scalp', 'bone','csf'      , 'gray'    , 'csf' , 'csf' , 'white'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   , 'tumor'   ], # noqa: E501
})

Tissue_params = pd.DataFrame({
    'spin_density':     [8000     , 1200      , 9800  , 8500  , 7450  , 6170  , 9500],  # noqa: E501
    't1_fitA':          [0.324    , 0.533     , None  , 1.35  , 0.857 , 0.583 , 0.926],  # noqa: E501
    't1_fitC':          [0.137    , 0.088     , None  , 0.34  , 0.376 , 0.382 , 0.217],  # noqa: E501
    't2':               [0.07     , 0.05      , 1.99  , 0.2   , 0.1   , 0.08  , 0.1],  # noqa: E501
    'chi':              [-7.5e-6  , -8.85e-6  , -9e-6 , -9e-6 , -9e-6 , -9e-6 , -9e-6],  # noqa: E501
    'type':             ['scalp'  ,'bone'     ,'csf'  ,'clot' ,'gray' ,'white', 'tumor']  # noqa: E501
})
# fmt: on
