from typing import Optional, Tuple

import torch

from ..aux.fourier import fourier_transform_adjoint, fourier_transform_forward
from ..aux.system import (
    system_matrix_adjoint,
    system_matrix_forward,
    system_matrix_normal,
    system_matrix_shape,
)
from . import LinearOperator


class SystemMatrixOperator(LinearOperator):
    """
    Operator for applying the system matrix in both forward and adjoint directions.
    Typically used in signal/image processing for structured data handling.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        window: Tuple[int, ...],
        axes: Tuple[int, ...] = (1, 2, 3),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the SystemMatrixOperator.

        Parameters:
        ----------
        input_shape : tuple of int
            Shape of the input tensor (nC, nX, nY, nZ, nS, nT1, nT2).
        window : tuple of int
            The size of the window in each dimension.
        axes : tuple of int, optional
            Axes along which the Fourier transform is applied (default is (1, 2, 3)).
        """
        nC, nX, nY, nZ, nS, nT1, nT2 = input_shape
        # Adjust window size based on input data shape
        window = (
            min(window[0], nX),
            min(window[1], nY),
            min(window[2], nZ),
        )
        nH1, nH2 = system_matrix_shape((nX, nY, nZ), window)

        self.forward_shape = (nC, nX, nY, nZ, nS, nT1, nT2)
        self.adjoint_shape = (nC, nH1, nH2, nS, nT1, nT2)

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        self.window = window
        self.axes = axes

        # Set the data types
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64
        self.device = device

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the forward system matrix operation.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The input tensor to be processed.

        Returns:
        -------
        torch.Tensor
            The result of applying the system matrix in the forward direction.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = fourier_transform_forward(input_tensor, self.axes)
        output_tensor = system_matrix_forward(
            output_tensor, self.forward_shape, self.window
        )
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the adjoint system matrix operation.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The input tensor to be processed in the adjoint direction.

        Returns:
        -------
        torch.Tensor
            The result of applying the system matrix in the adjoint direction.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = system_matrix_adjoint(
            input_tensor, self.forward_shape, self.window
        )
        output_tensor = fourier_transform_adjoint(output_tensor, self.axes)
        return output_tensor.ravel().view(self.dtype)


class SystemMatrixNormalOperator(LinearOperator):
    """
    Operator for applying the normal system matrix operation (forward and adjoint
    operations combined). This includes diagonal normalization, which is common in
    certain types of structured linear problems.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        window: Tuple[int, ...],
        axes: Tuple[int, ...] = (1, 2, 3),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the SystemMatrixNormalOperator.

        Parameters:
        ----------
        input_shape : tuple of int
            Shape of the input tensor (nC, nX, nY, nZ, nS, nT1, nT2).
        window : tuple of int
            The size of the window in each dimension.
        axes : tuple of int, optional
            Axes along which the Fourier transform is applied (default is (1, 2, 3)).
        """
        nC, nX, nY, nZ, nS, nT1, nT2 = input_shape
        # Adjust window size based on input data shape
        window = (
            min(window[0], nX),
            min(window[1], nY),
            min(window[2], nZ),
        )

        self.forward_shape = (nC, nX, nY, nZ, nS, nT1, nT2)
        self.adjoint_shape = (nC, nX, nY, nZ, nS, nT1, nT2)

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        # Create the diagonal for the normal system matrix operation
        self.diagonal = system_matrix_normal(
            torch.ones(input_shape, dtype=torch.complex64, device=device),
            input_shape,
            window,
        )

        self.axes = axes

        # Set the data types
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64
        self.device = device

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the normal system matrix operation in the forward direction.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The input tensor to be processed.

        Returns:
        -------
        torch.Tensor
            The result of the normal system matrix operation in the forward direction.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = fourier_transform_forward(input_tensor, self.axes)
        output_tensor = output_tensor * self.diagonal
        output_tensor = fourier_transform_adjoint(output_tensor, self.axes)
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the normal system matrix operation in the adjoint direction.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The input tensor to be processed in the adjoint direction.

        Returns:
        -------
        torch.Tensor
            The result of the normal system matrix operation in the adjoint direction.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = fourier_transform_forward(input_tensor, self.axes)
        output_tensor = output_tensor * self.diagonal
        output_tensor = fourier_transform_adjoint(output_tensor, self.axes)
        return output_tensor.ravel().view(self.dtype)
