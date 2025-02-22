from typing import Optional, Tuple

import torch

from ..aux.wavelet import (
    wavelet_transfer_functions,
    wavelet_transfer_functions_nd,
    wavelet_transform_adjoint,
    wavelet_transform_forward,
)
from . import LinearOperator


class WaveletTransformOperator(LinearOperator):
    """
    Operator for applying wavelet transforms in both forward and adjoint directions.
    Useful for multiscale signal processing and analysis.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        wavelet: str = "db2",
        level: int = 4,
        axes: Tuple[int, ...] = (0, 1),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the WaveletTransformOperator.

        Parameters:
        ----------
        input_shape : tuple of int
            Shape of the input tensor (nX, nY, nZ, nS, nTI, nTE).
        wavelet : str, optional
            Type of wavelet to use for the transformation (default is 'db2').
        level : int, optional
            Number of decomposition levels for the wavelet transform (default is 4).
        axes : tuple of int, optional
            Axes along which the wavelet transform is applied (default is (0, 1)).
        """
        # Number of dimensions for wavelet decomposition
        nX, nY, nZ, nS, nTI, nTE = input_shape
        nD = (3 * level) + 1

        # Calculate wavelet transfer functions for the axes
        Hx, Gx = wavelet_transfer_functions(
            wavelet, level, input_shape[axes[0]], device=device
        )
        Hy, Gy = wavelet_transfer_functions(
            wavelet, level, input_shape[axes[1]], device=device
        )

        # Combine the transfer functions for the specified axes
        self.F = wavelet_transfer_functions_nd(
            (Hx, Hy),
            (Gx, Gy),
            (nD,) + input_shape,
            axes,
            device=device,
        )

        self.axes = axes
        self.forward_shape = input_shape
        self.adjoint_shape = (nD,) + input_shape

        # Calculate the shape of the operator
        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        # Set the data types
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64
        self.device = device

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the wavelet transform in the forward direction.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The input tensor to be transformed.

        Returns:
        -------
        torch.Tensor
            The result of applying the wavelet transform in the forward direction.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = wavelet_transform_forward(
            input_tensor, self.F, self.axes
        ).clone()
        return output_tensor.view(self.dtype).ravel()

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the wavelet transform in the adjoint direction.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The input tensor to which the adjoint wavelet transform is applied.

        Returns:
        -------
        torch.Tensor
            The result of applying the wavelet transform in the adjoint direction.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = wavelet_transform_adjoint(
            input_tensor, self.F, self.axes
        ).clone()
        return output_tensor.view(self.dtype).ravel()
