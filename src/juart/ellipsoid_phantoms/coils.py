import time
from typing import List, Optional, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


class Coil:
    "Class of a simulated cylindric coil with channel arrays placed along z-axis."

    def __init__(
        self,
        coil_radius: float = 0.2,
        loop_radius: float = 0.04,
        num_loops_ring: int = 8,
        z_pos: Union[float, np.ndarray, list, tuple] = 0,
        phi0: Union[float, np.ndarray, list, tuple] = 0,
        z_orientation: Union[float, np.ndarray, list, tuple] = 0,
    ) -> object:
        """Init Coil.

        Parameters
        ----------
        coil_radius : float, optional
            Radius of the coil.
            Distance between z-axis and center of the coil channel loops) in [m],
            by default 0.2.
        loop_radius : float, optional
            Radius of the coil channel loops [m], by default 0.04
        num_loops_ring : int, optional
            Number of channels for each ring of channels, by default 8
        z_pos : float or np.ndarray or list or tuple, optional
            Positions of channel array rings along z axis [m], by default 0
            If scalar, only one channel array is created at z=0.
        phi0 : float or np.ndarray or list or tuple, optional
            Rotation shift of single channel arrays with phi0 [rad], by default 0
            If scalar, only one channel array is created.
        z_orientation : float or np.ndarray or list or tuple, optional
            Orientation of single channel arrays along z axis [rad], by default 0
            If scalar, only one channel array is created.
        """
        self.coil_radius = coil_radius
        self.loop_radius = loop_radius
        self.num_loops_ring = num_loops_ring
        self.type = "Rings"
        self.verbose = 0

        self.coeff_sig = None
        self.coeff_k = None

        # create coil objects
        z_pos_temp, phi0_temp, z_ori_temp = _modify_parameters(
            z_pos, phi0, z_orientation
        )
        phi_ring = np.linspace(0, 2 * np.pi, self.num_loops_ring, endpoint=False)
        self.coil_loops = []
        for z, p0, z_ori in zip(z_pos_temp, phi0_temp, z_ori_temp):
            for phi in phi_ring:
                phi += p0
                coil_normal = np.array([np.cos(phi), np.sin(phi), z_ori])
                coil_center = np.array(
                    [self.coil_radius * np.cos(phi), self.coil_radius * np.sin(phi), z]
                )
                self.coil_loops.append(
                    CoilLoop(coil_center, coil_normal, R=self.loop_radius)
                )

        self.num_channels = len(self.coil_loops)

    @property
    def coil_channel_positions(self):
        """Get the coil channel positions."""
        return np.array([coil.r_cent for coil in self.coil_loops])

    def get_coil_sens(self, r, verbose=0):
        """Simulates coil sensitivities at positions `r`.

        Parameters
        ----------
        r : np.ndarray, (3, N)
            Locations where coil sensitivities should be calculated [m].
        verbose: int
            If 1, print the time taken to simulate coil sensitivities.

        Returns
        -------
        S : (num_channels, N) complex ndarray
            Coil sensitivities for `num_channels` of the coil object.
        """
        t0 = time.time()
        num_dim, num_samples = r.shape

        S = np.empty((self.num_channels, num_samples), dtype=np.complex64)

        for n_cha in range(self.num_channels):
            S[n_cha] = self.coil_loops[n_cha].get_sensitivity(r)

        t1 = time.time()

        if verbose == 1:
            print(
                "Simulated coil sensitivities",
                f"for {self.num_channels} channels in {round(t1 - t0, 4)}s",
            )

        return S

    def get_sens_maps(self, matrix: npt.ArrayLike, fov: npt.ArrayLike, verbose=0):
        """Simulates coil sensitivity map.

        Parameters
        ----------
        matrix : np.ndarray, (D,)
            Matrix size for the coil sensitivity map in D dimensions.
        fov : np.ndarray, (D, )
            Field of view of the coil sensitivity map in D dimensions.

        Returns
        -------
        S : (self.num_channels, Nx, Ny, Nz) complex ndarray
            Coil sensitivity map for the coil object.
            If D=2, shape is (self.num_channels, Nx, Ny, 1).
            If D=3, shape is (self.num_channels, Nx, Ny, Nz).
        """

        sens_maps = []
        for n_cha in range(self.num_channels):
            sens_maps.append(self.coil_loops[n_cha].get_sensitivity_map(matrix, fov))
        sens_maps = np.concatenate(sens_maps, axis=0)

        return sens_maps

    def get_coil_sens_ksp(
        self,
        matrix: npt.ArrayLike,
        fov: npt.ArrayLike,
        L: Optional[List[int]] = None,
        verbose=0,
    ) -> np.ndarray:
        """Get coil sensitivity coefficients in kspace for grid location L in kspace.

        Parameters
        ----------
        matrix : npt.ArrayLike, (3,)
            Matrix size in kspace to calculate the coil sensitivities.
        fov : npt.ArrayLike, (3,)
            Field of view in which to calcualte the coil sensitivitiy coefficients.
        L : npt.ArrayLike, (3,)
            Matrix size in kspace to calculate the coil sensitivity coefficients.
            Usually use (6, 6, 6) for 3D.
        verbose : int, optional
            _description_, by default 0

        Returns
        -------
        sens_coeff : np.ndarray, Shape (num_channels, *L)
            Coil coefficients in kspace.
        """
        # TODO2 : This should just loop through the
        # coil loops and get the coil sensitivities
        # To do so implement TODO2.1
        matrix = np.asarray(matrix)
        fov = np.asarray(fov)

        # Create grid in image space
        x, y, z = [np.linspace(-fov[i] / 2, fov[i] / 2, matrix[i]) for i in range(3)]
        grid = np.meshgrid(x, y, z, indexing="ij")

        # Vectorize the grid
        r_3d = np.stack(
            [grid[0].flatten(), grid[1].flatten(), grid[2].flatten()], axis=0
        )

        # Create grid in kspace
        if L is None:
            L = [6, 6, 6]

        L = np.array(L)

        kmax = 1 / (2 * fov) * L // 2

        kx, ky, kz = [np.linspace(-kmax[i] / 2, kmax[i] / 2, L[i]) for i in range(3)]
        ksp_grid = np.meshgrid(kx, ky, kz, indexing="ij")

        # Vectorize the grid in kspace
        k_3d = np.stack(
            [ksp_grid[0].flatten(), ksp_grid[1].flatten(), ksp_grid[2].flatten()],
            axis=0,
        )

        # Get coil sensitivities coefficients
        sens_k = self.get_coil_sens_kcoeff(r_3d, k_3d, verbose=verbose)

        # Reshape back to grid
        sens_k = sens_k.reshape((self.num_channels, *L))

        return sens_k

    def get_coil_sens_kcoeff(
        self, r: np.ndarray, k: np.ndarray, verbose=0
    ) -> np.ndarray:
        """Calculates and fits the sensitivities
        at positions `r` to kspace locations `k`.

        Parameters
        ----------
        r : np.ndarray, Shape (3, N)
            Image space locations in [m].
        k : np.ndarray, Shape (3, M)
            Kspace locations in [m^-1].
        verbose : int, optional
            Information printing, by default 0

        Returns
        -------
        np.ndarray, (C, M)
            Fitted coil sensitivities coefficients to kspace locations.
        """
        # Get coil sensitivities
        sens = self.get_coil_sens(r, verbose=verbose)

        sens_k = _fit_coil_sens(r, sens, k, verbose=verbose)

        return sens_k


class CoilLoop:
    """Class for a single, circular coil loop.
    The loop consist of `n_phi' straight elements."""

    def __init__(
        self,
        r_cent: np.ndarray,
        normal: np.ndarray,
        R: float = 0.05,
        num_dphi: int = 50,
    ):
        """Create the coil loop object.

        Parameters
        ----------
        r_cent : np.ndarray, (3, )
            Distance vector to the center of the coil loop.
        normal : np.ndarray (3, )
            Normal vector of the surface of the coil loop.
        R : float, optional
            Radius of coil loop in [m]., by default 0.05
        num_dphi : int, optional
            Number of straight elements in the loop, by default 50
        """
        self.R = R
        self.phi, self.dphi = np.linspace(
            0, 2 * np.pi, num_dphi, endpoint=False, retstep=True
        )
        self.r_cent = r_cent
        self.normal = normal / np.linalg.norm(normal)

        # Calculate vector from center to coil elements and the dl vectors
        self.r_coil_elements, self.dl_coil_elements = self._build_coil_elements()

    def _build_coil_elements(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build coil elements of the loop.
        Reuturns the r (3, num_dphi) and dl (3, num_dphi) vectors of the coil elements.
        """
        # For easier computation, we define the coil loop normal vector to point
        # in x-direction and then rotate it to the desired normal vector
        NORMAL0 = np.array([1, 0, 0])

        rphi0 = np.array(
            [[0, np.cos(p) * self.R, np.sin(p) * self.R] for p in self.phi]
        )
        dl0 = np.array([[0, -np.sin(p) * self.R, np.cos(p) * self.R] for p in self.phi])

        q = _find_quaternion(NORMAL0, self.normal)
        r = Rotation.from_quat(q)

        norm_temp = r.apply(NORMAL0)

        if not np.equal(np.round(norm_temp, 4), np.round(self.normal, 4)).all():
            raise ValueError("Rotation of coil could not be estimated")

        r_coil_elements = self.r_cent + r.apply(rphi0)
        dl_coil_elements = r.apply(dl0)

        # Make dimension axis first axis to get (3, N) shape
        r_coil_elements = r_coil_elements.T
        dl_coil_elements = dl_coil_elements.T

        return r_coil_elements, dl_coil_elements

    def get_sensitivity(self, r: np.ndarray) -> np.ndarray:
        """Calculate complex sensitivity signal at r by biot savart law
        Parameters
        ----------
        r : ndarray, shape (3, num_samples)
            Observation points in [m]

        Returns
        -------
        ndarray, (num_samples, )
            Complex sensitivity signal at observation points `r`.
        """

        # calculate biot savart
        B = _biot_savart(r, self.r_coil_elements, self.dl_coil_elements, self.dphi)

        S = B[0] - 1j * B[1]

        # S /= np.max(np.abs(S))

        return S

    def get_sensitivity_map(
        self,
        matrix: npt.ArrayLike,
        fov: npt.ArrayLike,
    ) -> np.ndarray:
        """Simulates coil sensitivity map.

        Parameters
        ----------
        matrix : np.ndarray, (D,)
            Matrix size for the coil sensitivity map in D dimensions.
        fov : np.ndarray, (D, )
            Field of view of the coil sensitivity map in D dimensions.

        Returns
        -------
        S : (1, Nx, Ny, Nz) complex ndarray
            Coil sensitivity map for the coil object.
            If D=2, shape is (1, Nx, Ny, 1).
            If D=3, shape is (1, Nx, Ny, Nz).
        """
        matrix = list(matrix)
        fov = list(fov)

        if len(matrix) < 3:
            matrix = matrix + [1] * (3 - len(matrix))
        if len(fov) < 3:
            fov = fov + [0] * (3 - len(fov))

        # Get vectorized sample locations
        x, y, z = [np.linspace(-fov[i] / 2, fov[i] / 2, matrix[i]) for i in range(3)]
        grid = np.meshgrid(x, y, z, indexing="ij")

        r_3d = np.stack(
            [grid[0].flatten(), grid[1].flatten(), grid[2].flatten()], axis=0
        )

        # Get coil sensitivities
        sensitivities = self.get_sensitivity(r_3d)

        # Reshape back to grid
        sens_map = sensitivities.reshape((1, *matrix))

        # Scale to max
        sens_map /= np.max(np.abs(sens_map))

        return sens_map

    # TODO1.1 why not also create a function to get the coil sensitivity maps?

    # TODO2.1 why not also create a function to get coil sens in kspace?

    # TODO Let Coil just call the upper todos and looping through the coil channels
    # This would shift the complexity
    # to the CoilLoop class (because thats where the sensitivitites come from)
    # and make the Coil class more readable and easier to understand


def _fit_coil_sens(
    r: np.ndarray, sens: np.ndarray, k: np.ndarray, verbose: int = 0
) -> np.ndarray:
    """Fit the coil sensitivities `sens`
    from image locations `r` to kspace locations `k`.

    Parameters
    ----------
    r : np.ndarray, Shape (3, N)
        Image space locations in [m].
    sens : np.ndarray, Shape (C, N)
        Coil sensitivities at locations `r` with C channels.
    k : np.ndarray, Shape (3, M)
        Kspace locations in [m^-1].
    verbose : int, optional
        Information output, by default 0

    Returns
    -------
    np.ndarray, Shape (C, M)
        Fitted coil sensitivities to kspace locations.

    Raises
    ------
    ValueError
        If the number of samples in `r` and `sens` are not the same.
    ValueError
        If number of dimensions in `r` and `k` are not the same.
    """
    if sens.shape[1] != r.shape[1]:
        raise ValueError("Number of samples in r and sens must be the same.")

    if r.shape[0] != k.shape[0]:
        raise ValueError("Number of dimensions in r and k must be the same.")

    t0 = time.time()
    # Fit the sensitivity signals from image space to kspace
    A = np.exp(1j * 2 * np.pi * np.dot(r.T, k))
    sens_k, res, rank, sing = np.linalg.lstsq(A, sens.T, rcond=None)

    t1 = time.time()
    if verbose == 1:
        rmse = np.sqrt(res)
        nrmse = rmse / np.linalg.norm(sens, axis=1)

        print(
            "Fitted coil sensitivities to kspace locations ",
            f"in {round(t1 - t0, 4)}s with NRMSE: {nrmse}",
        )

    return sens_k


def _biot_savart(
    r: np.ndarray, rl: np.ndarray, dl: np.ndarray, dphi: float
) -> np.ndarray:
    """Calculates the Biot-Savart law for wire elements dl at locations rl for point r.

    Parameters
    ----------
    r : np.ndarray, Shape (3, num_samples)
        Observation points in [m]
    rl : np.ndarray, Shape (3, num_dphi)
        Locations of the wire elements in [m]
    dl : np.ndarray, Shape (3, num_dphi)
        Differential elements of the wire in [m].
    dphi : float
        Differential angle element

    Returns
    -------
    np.ndarray
        Magnetic field at observation points, shape (3, num_samples)

    The Biot-Savart law is given by:

    .. math::
        \\mathbf{B}(\\mathbf{r}) = \\frac{\\mu_0}{4\\pi} \\int \\frac{d\\mathbf{l}
        \\times (\\mathbf{r} - \\mathbf{r'})}{|\\mathbf{r} - \\mathbf{r'}|^3}

    where:
    - \\mathbf{B}(\\mathbf{r}) is the magnetic field at point \\mathbf{r}
    - \\mu_0 is the permeability of free space
    - d\\mathbf{l} is the differential length element of the wire
    - \\mathbf{r'} is the position vector of the differential element
    - \\mathbf{r} is the position vector where the field is being calculated
    """
    num_dim, num_samples = r.shape
    _, num_dphi = rl.shape

    B = np.zeros((num_dim, num_samples), dtype=np.complex64)
    for i in numba.prange(num_dphi):
        r_diff = r - rl[:, i][:, np.newaxis]
        cross = np.cross(dl[:, i], r_diff, axisb=0).T
        norm = np.linalg.norm(r_diff, axis=0) ** 3
        B += cross / norm * dphi

    return B


def _find_quaternion(v1, v2) -> np.ndarray:
    """Calculate the quaternion which rotate vector v1 (3,) onto vector v2 (3,).
    Returns the quaternion as a (4,) vector."""
    # Calculate the cross product of v1 and v2
    a = np.cross(v1, v2)

    # Calculate the components of the quaternion q
    q = np.zeros(4)  # Initialize the quaternion as [0, 0, 0, 0]

    # Set the xyz components of q to the calculated cross product (a)
    q[0:3] = a

    # Calculate the scalar component (w) of the quaternion
    q[3] = np.sqrt(np.linalg.norm(v1) ** 2 * np.linalg.norm(v2) ** 2) + np.dot(v1, v2)

    return q


def _modify_parameters(
    z_pos: Union[float, np.ndarray, list, tuple],
    phi0: Union[float, np.ndarray, list, tuple],
    z_orientation: Union[float, np.ndarray, list, tuple],
) -> tuple:
    """
    Modify the input parameters as follows:
        z_pos is the main parameter. It can be a scalar value or array-like.
        If z_pos is a scalar, the other two parameters must not be array-like,
        but also scalars.
        If z_pos is array-like, the other two parameters can be scalars and are
        tiled to match the length of z_pos.
        If z_pos is array-like and the other parameters are also array-like,
        they must have the same length as z_pos.

    Parameters
    ----------
    z_pos : float or np.ndarray or list or tuple
        Position parameter.
    phi0 : float or np.ndarray or list or tuple
        Phi0 parameter.
    z_orientation : float or np.ndarray or list or tuple
        Orientation parameter.

    Returns
    --------
    tuple: Tuple containing modified z_pos, phi0, and z_orientation as lists.
    """

    # Handle z_pos
    if isinstance(z_pos, (float, int)):
        z_pos = [z_pos]
    else:
        z_pos = list(z_pos)

    # Handle phi0
    if isinstance(phi0, (float, int)):
        phi0 = [phi0] * len(z_pos)
    else:
        if len(phi0) != len(z_pos):
            raise ValueError("phi0 should have the same length as z_pos.")
        phi0 = list(phi0)

    # Handle z_orientation
    if isinstance(z_orientation, (float, int)):
        z_orientation = [z_orientation] * len(z_pos)
    else:
        if len(z_orientation) != len(z_pos):
            raise ValueError("z_orientation should have the same length as z_pos.")
        z_orientation = list(z_orientation)

    return z_pos, phi0, z_orientation
