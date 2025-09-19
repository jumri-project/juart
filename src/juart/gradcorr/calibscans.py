# As the gradient calibration does not need to be differentiable and does not compute
# large matrix operations, all internal calculations are performed with numpy on cpu.
import os
from typing import Literal, Optional, Tuple
import numpy as np
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

class CorrectionScan:
    r"""
    Class to calculate the gradient delay of a single axis from calibration scan data.
    """

    def __init__(
        self,
        ktraj,
        sig,
        axis: Literal["x", "y", "z"],
        window: int = 30,
    ):
        r"""
        Parameters
        ----------
        ktraj : np.ndarray, shape (1, N, 2),
            K-space trajectory.
        sig : np.ndarray, shape (C, N, 2),
            Magnitude signal.
        
        """
        self.window = window
        self.axis = axis
        
        # Find zero crossings in trajectory
        zero_crossings = _find_zero_crossings(ktraj=ktraj)

        # Filter crossings
        limits = (0, ktraj.shape[1])
        self.zero_crossings = _window_selection(zero_crossings, window, limits)

        # Get trajectory and signal intervalls
        intervall_borders = [(x-window//2, x+window//2) for x in self.zero_crossings]
        k_intervalls = [ktraj[:, slice(itv[0], itv[1]), :] for itv in intervall_borders]
        d_intervalls = [sig[:, slice(itv[0], itv[1]), :] for itv in intervall_borders]

        # Interpolate signal to equidistant trajectory samples
        k_interp, d_interp = _interpolate_intervalls(k_intervalls, d_intervalls)

        # Calculate cross spectrum
        g = _cross_spectrum(d_interp[..., 0], d_interp[..., 1])
        

    

class GradCorrection:
    def __init__(
        self,
    ):
        pass


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
    x_limits: Optional[Tuple[int, int]] = None,
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
    prev = np.concatenate(
        [np.array([x_limits[0]]), x_sort[:-1]]
    )
    nxt = np.concatenate(
        [x_sort[1:], np.array([x_limits[1]])]
    )

    # Estimate distance to neighbors
    dist_left = x_sort - prev
    dist_right = nxt - x_sort

    # Create mask from distances
    keep_mask = (dist_left >= half_w) & (dist_right >= half_w)

    return x_sort[keep_mask]

def _interpolate_signal(
    ktraj: np.ndarray, signal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
    ktraj_interpolate = np.arange(kmin, kmax+0.5)

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
        ktraj_interpolate[None, :, None],
        (1, ktraj_interpolate.size, num_echo)
    )

    return ktraj_interpolate, signal_interpolate

def _interpolate_intervalls(
        k_list: list[np.ndarray], d_list: list[np.ndarray]
    ) -> Tuple[list[np.ndarray], list[np.ndarray]]:
            """Return the interpolated signal and trajectory data
            for each element of d_list and k_list."""
            k_list_out, d_list_out = [], []

            for k, d in zip(k_list, d_list):
                k_interp, d_interp = _interpolate_signal(k, d)
                k_list_out.append(k_interp)
                d_list_out.append(d_interp)

            return k_list_out, d_list_out

def _cross_spectrum(x: np.ndarray, y :np.ndarray) -> np.ndarray:
    """Calculate the cross spectrum of x and y. x and y must have shape (C, M)
    and the cross spectrum is calculated along M"""
    num_cha, num_samples = x.shape

    fft_1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))
    fft_2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y)))

    g = fft_1 * np.conjugate(fft_2)

    return g

    