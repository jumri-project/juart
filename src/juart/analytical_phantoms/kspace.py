import numba
import numpy as np
import pandas as pd
from scipy.special import j1


def ellipseFT(k: np.ndarray, params: pd.DataFrame) -> np.ndarray:
    """Calculate signal for single ellipsoid with density 1
    at k-space locations k described in [1].
    Calculation is possible for 3D non selective rf-pulse
    or 2D slice-selective rf-pulse by dimension of k.

    Parameters
    ----------
    k : np.ndarray
        K-space locations of shape (Nsamples, Ndim) [1/m].
    params : pd.DataFrame
        Shape parameters of ellipsoid in [m] and [degree].
        Keys 3D: [center_x, center_y, center_z, axis_a, axis_b, axis_c, angle]

    Returns
    -------
    np.ndarray
        Complex signal for k-space locations of shape (Nsamples,).

    References
    ----------
    ..[1] Koay, C.G., Sarlls, J.E., Özarslan, E., 2007.
        "Three-dimensional analytical magnetic resonance imaging
        phantom in the Fourier domain" MRM.
        https://doi.org/10.1002/mrm.21292
    """

    Nsamples, Ndim = k.shape

    ax, ct, R = _extract_params(params, Ndim)

    # extract centers, axes and rotation angle
    if Ndim == 2:
        k_signal = _core_ellipse_FT_2D(k, ct, ax, R)

    elif Ndim == 3:
        k_signal = _core_ellipse_FT_3D(k, ct, ax, R)

    return k_signal


def ellipseFT_sens(
    k: np.ndarray, params: np.ndarray, k_sens: np.ndarray, s_sens: np.ndarray
) -> np.ndarray:
    """Calculate signal for single ellipsoid with density 1
    at k-space locations k like described in [1] together with
    coil sensitivities.
    Calculation is possible for 3D non selective rf-pulse
    or 2D slice-selective rf-pulse by dimension of k.

    Parameters
    ----------
    k : np.ndarray
        K-space locations of shape (Nsamples, Ndim) [1/m].
    params : np.ndarray
        Shape parameters of ellipsoid in [m] and [degree].
        3D: [center_x, center_y, center_z, axis_a, axis_b, axis_c, angle]
        2D: [center_x, center_y, axis_a, axis_b, angle]
    k_sens : np.ndarray
        K-space location of coil sensitivity coefficients (Ncoeff, Ndim) [1/m]
    s_sens : np.ndarray, np.complex64
        Fourier transform of coil sensitivity coefficients (Ncoeff, )

    Returns
    -------
    np.ndarray
        Complex signal for k-space locations of shape (Nsamples,).

    References
    ----------
    ..[1] Koay, C.G., Sarlls, J.E., Özarslan, E., 2007.
        "Three-dimensional analytical magnetic resonance imaging
        phantom in the Fourier domain". MRM.
        https://doi.org/10.1002/mrm.21292
    """

    # check if k is of right shape
    if not ((k.ndim == 2) and (k.shape[1] == 3 or k.shape[1] == 2)):
        raise ValueError(
            f"Dimensions of k should be (NSamples, 2) \
            or (Nsamples, 3) but are {k.shape}."
        )

    Nsamples, Ndim = k.shape

    ax, ct, R = _extract_params(params, Ndim)

    return _coil_convolve(k, k_sens, s_sens, ct, ax, R)


"""
# It seems like numba calculation of dot product and norm is quite shitty
# for the standard numpy.dot and numpy.linalg.norm functions
# Also, the taylor approximation is different for 3d and 2d.
# Thats why in each function a check for 2d or 3d is needed at the moment
"""


@numba.njit(parallel=True)
def _coil_convolve(k, k_sens, s_sens, ct, ax, R):
    Nsamples, Ndim = k.shape
    Ncoeffs = k_sens.shape[0]

    output = np.zeros(Nsamples, dtype=np.complex64)
    if Ndim == 2:
        for i in numba.prange(Ncoeffs):
            w = k - k_sens[i]
            output += _core_ellipse_FT_2D(w, ct, ax, R) * s_sens[i]

    elif Ndim == 3:
        for i in numba.prange(Ncoeffs):
            w = k - k_sens[i]
            output += _core_ellipse_FT_3D(w, ct, ax, R) * s_sens[i]

    return output


@numba.njit(parallel=True)
def _core_ellipse_FT_2D(k, ct, ax, rot):
    """Core function for computation of ellipsoid ft"""
    exp = np.exp(-1j * 2 * np.pi * (k[:, 0] * ct[0] + k[:, 1] * ct[1]))

    kappa = np.sqrt(
        (
            (ax[0] * (k[:, 0] * rot[0, 0] + k[:, 1] * rot[1, 0])) ** 2
            + (ax[1] * (k[:, 0] * rot[0, 1] + k[:, 1] * rot[1, 1])) ** 2
        )
    )

    # calculcation of bessel function
    zeros = kappa < 0.001

    k_bessel = np.empty(kappa.shape[0])

    k_bessel[~zeros] = j1_numba(2 * np.pi * kappa[~zeros]) / (np.pi * kappa[~zeros])

    k_bessel[zeros] = (
        1 - 1 / 2 * (np.pi * kappa[zeros] ** 2) + 1 / 12 * (np.pi * kappa[zeros]) ** 4
    )

    sig = np.pi * np.prod(ax) * exp * k_bessel
    return sig


@numba.njit(parallel=True)
def _core_ellipse_FT_3D(k, ct, ax, rot):
    """Core function for computation of 3d ellipsoid ft"""

    exp = np.exp(
        -1j * 2 * np.pi * (k[:, 0] * ct[0] + k[:, 1] * ct[1] + k[:, 2] * ct[2])
    )

    kappa = np.sqrt(
        (ax[0] * (k[:, 0] * rot[0, 0] + k[:, 1] * rot[1, 0])) ** 2
        + (ax[1] * (k[:, 0] * rot[0, 1] + k[:, 1] * rot[1, 1])) ** 2
        + (ax[2] * k[:, 2]) ** 2
    )

    # calculcation of bessel function
    zeros = kappa < 0.001

    k_bessel = np.empty(kappa.shape[0])

    k_bessel[~zeros] = (
        np.sin(2 * np.pi * kappa[~zeros])
        - 2 * np.pi * kappa[~zeros] * np.cos(2 * np.pi * kappa[~zeros])
    ) / (2 * np.pi**2 * kappa[~zeros] ** 3)

    k_bessel[zeros] = (
        4 / 3 * np.pi
        - 8 / 15 * np.pi**3 * kappa[zeros] ** 2
        + 8 / 105 * np.pi**5 * kappa[zeros] ** 5
    )

    sig = np.prod(ax) * exp * k_bessel

    return sig


def _extract_params(params, Ndim):
    """Extract and modify parameters from ellipsis

    Parameters
    ----------
    params : np.ndarray
        Shape parameters of ellipsoid in [m] and [degree].
        3D: [center_x, center_y, center_z, axis_a, axis_b, axis_c, angle]
        2D: [center_x, center_y, axis_a, axis_b, angle]
    Ndim : int
        Dimension of ellipse. 2D or 3D.

    Returns
    -------
    ax : np.ndarray
        Axes of ellipsoid. [axis_a, axis_b], [axis_a, axis_b, axis_c]
    ct : np.ndarray
        Center of ellipsoid. [center_x, center_y], [center_x, center_y, center_z]
    R : np.ndarray
        Rotation matrix for the rotation of the ellipsoid around x in 2D and z in 3D.
    """
    if Ndim == 3:
        if params.shape[0] != 7:
            raise ValueError(
                f"Number of parameters not sufficient"
                f"Expected 7 parameters for {Ndim} ellips but got {params.shape[0]}."
                f"Parameters have to be [center_x, center_y, center_z, \
                axis_a, axis_b, axis_c, angle]"
                f"Got parameters {params}"
            )

        xc, yc, zc, xa, ya, za, ang = params

        ang = np.deg2rad(ang)

        ax = np.array([xa, ya, za])
        ct = np.array([xc, yc, zc])

        R = np.array(
            [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
        )

    elif Ndim == 2:
        if params.shape[0] != 5:
            raise ValueError(
                f"Number of parameters not sufficient"
                f"Expected 5 parameters for {Ndim} ellips but got {params.shape[0]}."
                f"Parameters have to be [center_x, center_y, axis_a, axis_b, angle]"
                f"Got parameters {params}"
            )

        xc, yc, xa, ya, ang = params

        ang = np.deg2rad(ang)

        ax = np.array([xa, ya])
        ct = np.array([xc, yc])

        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    else:
        raise ValueError(f"Expected 2 or 3 dimensional ellips but got {Ndim}")

    return ax, ct, R


@numba.vectorize([numba.float64(numba.float64)])
def j1_numba(x):
    """Just for numba use of the scipy.special.j1 function"""
    return j1(x)


def add_noise(signal: np.ndarray, k: np.ndarray, snr_db: float) -> np.ndarray:
    r"""Function to add noise to signal depending on desired signal to noise ratio.

    Parameters
    ----------
    signal : :math:'(N, M)' np.ndarray
        Complex signal with :math:'N' samples and :math:'M' coil channels.

    k : :math:'(N, K)' np.ndarray
        K-space locations with N samples and K dimensions.

    snr_db : float
        Signal to noise ratio [dB].
        Following Nishimura "Principles of Magnetic Resonance Imaging"
        :math:'SNR_{dB} = 20 \log(SNR)'

    Returns
    -------
    signal
        Input signal added with noise.
    """
    # for an snr of over 50dB there is effectivly no noise
    if snr_db >= 40:
        return signal

    # We do not want to do math with snr in dB
    snr = 10 ** (snr_db / 20)

    # Compute distance of kspace locations to center of kspace
    k_abs = np.linalg.norm(k, axis=1)

    # Find the k-space locations which are not more than
    # 10% away of the center than the maximum k-space location distance
    ind = np.where((k_abs < 0.1 * np.max(k_abs)))[0]

    # Extract corresponding rows from m
    signal_sub = signal[ind, :]

    # Generate complex Gaussian noise
    n = np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)

    # Center the noise to be realy at mu=0
    n = n - np.tile(np.mean(n, axis=0), (signal.shape[0], 1))

    # Scale the noise to achieve the desired SNR
    n = np.mean(np.abs(signal_sub), axis=0) / snr * (n / np.std(n))

    # Add noise to signal
    signal_mod = signal + n

    return signal_mod
