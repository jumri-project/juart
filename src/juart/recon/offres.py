import numpy as np


class OffResonanceCorrection:
    """Class for time segmentation for off-resonance correction.
    Based on Noll et al.,
    "A homogeneity correction method for magnetic resonance imaging
    with time-varying gradients", IEEE, 1991
    """

    def __init__(
        self, B0_map: np.ndarray, num_seg: int, num_samples: int, dwell: float
    ):
        """Initialize the OffResonanceCorrection class.

        Parameters
        ----------
        B0_map : np.ndarray, (1, Nx, Ny, Nz)
            B0 field map with 1 channel.
        num_seg : int
            Number of time segments to use for the reconstruction.
        num_samples : int
            Number of samples in the readout.
        dwell : float
            Dwell time of readout in seconds.
        """
        self.B0_map = B0_map
        self.num_seg = num_seg
        self.num_samples = num_samples
        self.window_width = num_samples / (num_seg - 1)
        self.dwell = dwell

    def get_signal_weights(self, n_seg: int):
        """Return the weights in for kspace signal for the given segment `n_seg'.

        Parameters
        ----------
        n_seg : int
            Number of the segment to get the weights for.

        Returns
        -------
        weights : np.ndarray, (1, `num_samples`)
            Weights for kspace samples for the given segment number
            `n_seg` with 1 channel and N=`num_samples` samples.
        """
        t = np.arange(self.num_samples)

        t0 = self.window_width * n_seg

        weights = 0.5 + 0.5 * np.cos(np.pi * (t - t0) / self.window_width)

        upper_bound = t0 + self.window_width
        lower_bound = t0 - self.window_width

        weights[t < lower_bound] = 0
        weights[t > upper_bound] = 0

        # Add channel dimension
        weights = weights[np.newaxis, :]

        return weights

    def get_img_phase(self, n_seg: int) -> np.ndarray:
        """Get the additional phase in image space
        for the given segment and slice of the B0 map.

        Parameters
        ----------
        n_seg : int
            Segment number.

        Returns
        -------
        np.ndarray, (1, Nx, Ny, Nz))
            Phase for the given segment of the B0 map.
        """

        t0 = self.window_width * n_seg * self.dwell
        phase = np.exp(1j * 2 * np.pi * self.B0_map * t0)

        return phase
