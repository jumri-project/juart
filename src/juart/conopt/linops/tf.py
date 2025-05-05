from typing import Optional, Tuple

import torch

from ..tfs import apply_oversampled_transfer_function, apply_transfer_function
from . import LinearOperator


class TransferFunctionNormalOperator(LinearOperator):
    """
    Operator for applying transfer functions in both forward and adjoint directions.
    Typically used in Fourier-based methods for structured data processing.
    """

    def __init__(
        self,
        transfer_function: torch.Tensor,
        input_shape: Tuple[int, ...],
        axes: Tuple[int, ...] = (1, 2),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the OversampledTransferFunctionNormalOperator.

        Parameters:
        ----------
        transfer_function : torch.Tensor
            Transfer function to apply in both forward and adjoint directions.
        input_shape : tuple of int
            Shape of the input tensor (nC, nX, nY, nZ, nS, nTI, nTE).
        axes : tuple of int, optional
            Axes along which the transfer function is applied (default is (1, 2)).
        """
        nC, nX, nY, nZ, nS, nTI, nTE = input_shape

        self.forward_shape = (nC, nX, nY, nZ, nS, nTI, nTE)
        self.adjoint_shape = (nC, nX, nY, nZ, nS, nTI, nTE)

        self.axes = axes

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        self.transfer_function = transfer_function.to(device)

        # Set the data types
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64
        self.device = device

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the oversampled transfer function in the forward direction.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to which the forward transfer function is applied.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the forward transfer function.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = apply_transfer_function(
            input_tensor, self.transfer_function, self.axes
        )
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the oversampled transfer function in the adjoint direction.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to which the adjoint transfer function is applied.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the adjoint transfer function.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = apply_transfer_function(
            input_tensor, self.transfer_function, self.axes
        )
        return output_tensor.ravel().view(self.dtype)


class OversampledTransferFunctionNormalOperator(LinearOperator):
    """
    Operator for applying oversampled transfer functions in both forward and adjoint
    directions. Typically used in Fourier-based methods for structured data processing.
    """

    def __init__(
        self,
        transfer_function: torch.Tensor,
        input_shape: Tuple[int, ...],
        nonuniform_axes: Tuple[int, ...] = (1, 2),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the OversampledTransferFunctionNormalOperator.

        Parameters:
        ----------
        transfer_function : torch.Tensor
            Transfer function to apply in both forward and adjoint directions.
        input_shape : tuple of int
            Shape of the input tensor (nC, nX, nY, nZ, nS, nTI, nTE).
        nonuniform_axes : tuple of int, optional
            Axes along which the transfer function is applied (default is (1, 2)).
        """
        nC, nX, nY, nZ, nS, nTI, nTE = input_shape

        self.forward_shape = (nC, nX, nY, nZ, nS, nTI, nTE)
        self.adjoint_shape = (nC, nX, nY, nZ, nS, nTI, nTE)

        self.nonuniform_axes = nonuniform_axes

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        self.transfer_function = transfer_function.to(device)

        # Set the data types
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64
        self.device = device

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the oversampled transfer function in the forward direction.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to which the forward transfer function is applied.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the forward transfer function.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = apply_oversampled_transfer_function(
            input_tensor, self.transfer_function, self.nonuniform_axes
        )
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the oversampled transfer function in the adjoint direction.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to which the adjoint transfer function is applied.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the adjoint transfer function.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = apply_oversampled_transfer_function(
            input_tensor, self.transfer_function, self.nonuniform_axes
        )
        return output_tensor.ravel().view(self.dtype)
