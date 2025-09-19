# As the gradient calibration does not need to be differentiable and does not compute
# large matrix operations, all internal calculations are performed with numpy on cpu.
import math
from typing import Literal, Optional

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress


class CorrectionScan:
    r"""
    Class to calculate the gradient delay of a single axis from calibration scan data.
    """

    def __init__(
        self,
        ktraj: np.ndarray,
        sig: np.ndarray,
        dwell: float,
        axis: Literal["x", "y", "z"],
        window: int = 80,
        gamma: float = 42.576e6,
    ):
        r"""
        Parameters
        ----------
        ktraj : np.ndarray, shape (1, N, 2)
            K-space trajectory.
        sig : np.ndarray, shape (C, N, 2)
            Magnitude signal.
        dwell : float
            Sampling dwell time of the trajectory in seconds.
        axis : Literal["x", "y", "z"]
            The axis the correction scan was performed on.
        window : int, optional
            Size of the intervals where the trajectory delay will be calculated.
            Default is 80.
        gamma : float, optional
            Gyromagnetic ratio. Default is 42.576e6.
        """
        self.window = window
        self.axis = axis
        self.num_cha = sig.shape[0]

        # Get gradient waveform
        gtraj = np.gradient(ktraj, axis=1) / (dwell * gamma)

        # Find zero crossings in trajectory
        zero_crossings = _find_zero_crossings(ktraj=ktraj)

        # Filter crossings
        limits = (0, ktraj.shape[1])
        self.zero_crossings = _window_selection(zero_crossings, window, limits)
        self.num_crossings = self.zero_crossings.size

        # Check if we have any zero crossings to work and stop if we don't
        if self.num_crossings == 0:
            self._delays = np.zeros((self.num_crossings, self.num_cha))
            self._max_intensity = np.ones((self.num_crossings, self.num_cha))
            return

        # Get trajectory and signal intervalls
        intervall_borders = [
            (x - window // 2, x + window // 2) for x in self.zero_crossings
        ]
        k_intervalls = [ktraj[:, slice(itv[0], itv[1]), :] for itv in intervall_borders]
        d_intervalls = [sig[:, slice(itv[0], itv[1]), :] for itv in intervall_borders]

        # Get gradient amplitude for zero crossing
        gtraj_crossings = np.abs(gtraj[0, self.zero_crossings, 0])

        # Interpolate signal to equidistant trajectory samples
        _, d_interp = _interpolate_intervalls(k_intervalls, d_intervalls)

        # Calculate cross spectrum for each intervall
        cross_spec = [_cross_spectrum(d[:, :, 0], d[:, :, 1]) for d in d_interp]

        # Calculate shifts and maximum intensities of cross sections along channel dim
        shifts, max_intensity = zip(*[_cross_spectrum_delay(cs) for cs in cross_spec])
        shifts = np.array(shifts)
        max_intensity = np.array(max_intensity)

        # Convert shifts to seconds
        delays = shifts / (gtraj_crossings * gamma)

        self._delays = delays
        self._max_intensity = max_intensity

    def get_delay(
        self, crossing_limit: Optional[int] = None, weight_channel: bool = True
    ) -> float:
        r"""
        Return the estimated gradient delay.

        Parameters
        ----------
        crossing_limt: int or None
            Number of crossings to use for the estimation.
            If None, all crossings are used.
        weight_channel: bool
            If True, the mean is weighted by the maximum intensity of each crossing.

        Returns
        -------
        float
            Estimated gradient delay in seconds.
        """
        # Handle case of no crossings of kspace center
        if self.num_crossings == 0:
            return 0.0

        # Determine number of crossings to use
        if crossing_limit is None or crossing_limit > self.num_crossings:
            crossing_limit = self.num_crossings

        # Calculate weighted or unweighted mean of shifts
        if weight_channel:
            weights = (
                self._max_intensity[:crossing_limit]
                / self._max_intensity[:crossing_limit].sum()
            )
            output = (self._delays * weights).mean()
        else:
            output = self._delays[:crossing_limit].mean()

        return output


def _find_zero_crossings(ktraj: np.ndarray) -> np.ndarray:
    """Find the zero crossings of the trajectory and return their indices.

    Parameter
    ---------
    ktraj : np.array, shape (1, N)
        K-space trajectory for a single dimension.

    Returns
    -------
    crossings: np.array, shape (N, )
        Indices of the trajectory points where k=0.
    """
    # Make trajectory truly 1dim
    ktraj = ktraj.squeeze()

    # Find sign changes
    k_sign = np.sign(ktraj)

    # Handle the case where ktraj is exactly zero
    zero_ind = np.where(k_sign == 0)[0]
    k_sign[zero_ind] = k_sign[zero_ind + 1]

    # Find zero crossings
    crossings = np.where((k_sign[:-1] * k_sign[1:]) < 0)[0]

    return crossings


def _window_selection(
    x: np.ndarray,
    window=40,
    x_limits: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Filter a array of indices aso that each index is seperated
    from its neighbour by window/2.

    Parameters
    ----------
    x: np.array, shape (N, )
        Indices to filter.
    window : int
        Window width by which the indices should be separated.
    x_limits : Tuple(int, int)
        Limits of the domain where x is defined.

    Returns
    -------
    np.array, shape (M, )
        Filtered indices.
    """

    half_w = window // 2

    # Bring in ascending order
    x_sort = np.sort(x)

    # Create borders
    if x_limits is None:
        x_limits = (0, 2**63 - 1)

    # Add borders
    prev = np.concatenate([np.array([x_limits[0]]), x_sort[:-1]])
    nxt = np.concatenate([x_sort[1:], np.array([x_limits[1]])])

    # Estimate distance to neighbors
    dist_left = x_sort - prev
    dist_right = nxt - x_sort

    # Create mask from distances
    keep_mask = (dist_left >= half_w) & (dist_right >= half_w)

    return x_sort[keep_mask]


def _interpolate_signal(
    ktraj: np.ndarray,
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolates signal data to equisdistant point on the trajectory.

    Parameters
    ----------
    ktraj : np.ndarray, shape (1, N, 2)
        Trajectory data with 2 echoes.
    signal : np.ndarray, shape (C, N, 2)
        Signal magnitude data with C channels, N samples and 2 echoes.

    Returns
    -------
    ktraj_interpolate : np.ndarray (1, M, 2)
        Interpolated trajectory. Values are sorted in ascending order.
    signal_interpolate :np.ndarray (C, M, 2)
        Interpolated magnitude signal.
    """

    num_cha, _, num_echo = signal.shape

    # Find max and min trajectory
    kmin, kmax = ktraj.min(), ktraj.max()

    # Generate new, equidistant trajectory samples
    ktraj_interpolate = np.arange(kmin, kmax + 0.5)

    # Interpolate
    shape_signal_interp = (num_cha, ktraj_interpolate.size, num_echo)
    signal_interpolate = np.zeros(shape_signal_interp, dtype=np.float32)
    for n_echo in range(num_echo):
        k = ktraj[0, :, n_echo]
        s = signal[..., n_echo]

        if k[0] > k[1]:  # interpolation need increasing x
            k = k[::-1]
            s = s[:, ::-1]

        finterp = make_interp_spline(k, s, k=1, axis=1)

        signal_interpolate[..., n_echo] = finterp(ktraj_interpolate)

    # Return interpolated trajectory in same dimensions as input traj
    ktraj_interpolate = np.broadcast_to(
        ktraj_interpolate[None, :, None], (1, ktraj_interpolate.size, num_echo)
    )

    return ktraj_interpolate, signal_interpolate


def _interpolate_intervalls(
    k_list: list[np.ndarray],
    d_list: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return the interpolated signal and trajectory data
    for each element of d_list and k_list."""
    k_list_out, d_list_out = [], []

    for k, d in zip(k_list, d_list):
        k_interp, d_interp = _interpolate_signal(k, d)
        k_list_out.append(k_interp)
        d_list_out.append(d_interp)

    return k_list_out, d_list_out


def _cross_spectrum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the cross spectrum of x and y.
    Both x and y must have shape (C, M)
    and the cross spectrum is calculated along M"""
    fft_1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))
    fft_2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y)))

    g = fft_1 * np.conjugate(fft_2)

    return g


def _cross_spectrum_delay(
    g: np.ndarray, peak_tresh: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the delay from the cross spectrum g
    and return the cross spectrum maximum.

    Parameters
    ----------
    g : np.ndarray, complex, shape (C, M)
        Cross spectrum data.

    Returns
    -------
    delays : np.ndarray, shape (C,)
        Estimated delays in samples.
    max_intensitys : np.ndarray, shape (C,)
        Maximum intensitys of the cross spectrum.
    """
    num_cha, num_samples = g.shape

    # Find peaks in cross spectrum
    max_intensitys = np.zeros(num_cha)
    delays = np.zeros(num_cha)

    # Put values along positive axis
    x = np.arange(num_samples)

    for cha in range(num_cha):
        g_tmp = g[cha]

        # Select only center peak signal
        int_lim = np.abs(g_tmp).max() * peak_tresh
        mask = np.abs(g_tmp) >= int_lim
        g_inner = g_tmp[mask]
        x_inner = x[mask]

        if g_inner.size == 0:
            continue

        # Get phase
        g_phase = np.unwrap(np.angle(g_inner))

        # Calculate linear regression of unwrapped phase
        reg = linregress(x_inner, g_phase)

        # Slope is delay
        delay = -1 * reg.slope * num_samples / (2 * math.pi)

        delays[cha] = delay
        max_intensitys[cha] = int_lim

    return delays, max_intensitys
