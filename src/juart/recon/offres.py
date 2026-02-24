import torch


class OffResonanceCorrection:
    """Class for time segmentation for off-resonance correction.
    Based on Noll et al.,
    "A homogeneity correction method for magnetic resonance imaging
    with time-varying gradients", IEEE, 1991
    """

    def __init__(self, B0_map: torch.Tensor, num_seg: int, timestamps: torch.Tensor):
        """Initialize the OffResonanceCorrection class.

        Parameters
        ----------
        B0_map : torch.Tensor, (1, Nx, Ny, Nz)
            B0 field map in Hz.
        num_seg : int
            Number of time segments to use for the reconstruction.
        timestamps : torch.Tensor, (N,)
            Timestamps corresponding to the k-space samples, in seconds.
            Should have the same dwell time between samples.
        """
        self.B0_map = B0_map
        self.num_seg = num_seg
        self.timestamps = timestamps
        self.num_samples = len(timestamps)
        if self.num_samples < 2:
            raise ValueError(
                "At least two timestamps are required to calculate dwell time."
            )

        self.window_width = self.num_samples / (num_seg - 1)

        tmp = torch.sort(self.timestamps)[0]  # Ensure timestamps are sorted
        self.dwell = tmp[1] - tmp[0]

    def get_signal_weights(self, n_seg: int):
        """Return the weights in for kspace signal for the given segment `n_seg'.

        Parameters
        ----------
        n_seg : int
            Number of the segment to get the weights for.

        Returns
        -------
        weights : torch.Tensor, (1, `num_samples`)
            Weights for kspace samples for the given segment number
            `n_seg` with 1 channel and N=`num_samples` samples.
        """
        t0 = self.window_width * n_seg * self.dwell

        weights = 0.5 + 0.5 * torch.cos(
            torch.pi * (self.timestamps - t0) / (self.window_width * self.dwell)
        )

        upper_bound = t0 + self.window_width * self.dwell
        lower_bound = t0 - self.window_width * self.dwell

        weights[self.timestamps < lower_bound] = 0
        weights[self.timestamps > upper_bound] = 0

        # Add channel dimension
        weights = weights.unsqueeze(0)

        return weights

    def get_img_phase(self, n_seg: int) -> torch.Tensor:
        """Get the additional phase in image space
        for the given segment and slice of the B0 map.

        Parameters
        ----------
        n_seg : int
            Segment number.

        Returns
        -------
        torch.Tensor, (1, Nx, Ny, Nz)
            Phase for the given segment of the B0 map.
        """

        t0 = self.window_width * n_seg * self.dwell
        phase = torch.exp(1j * 2 * torch.pi * self.B0_map * t0)

        return phase
