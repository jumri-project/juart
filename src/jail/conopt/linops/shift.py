from typing import Tuple

import torch

from ..aux.shift import shift_adjoint, shift_forward
from . import LinearOperator


class ShiftOperator(LinearOperator):
    """
    Operator to apply forward and adjoint shift operations across multiple dimensions.
    """

    def __init__(
        self,
        shift_number: Tuple[int, ...],
        shift_size: Tuple[int, ...],
        axes: Tuple[int, ...],
        input_shape: Tuple[int, ...],
    ):
        """
        Initialize the ShiftOperator.

        Parameters:
        ----------
        shift_number : tuple of int
            The number of shifts along the specified axes.
        shift_size : tuple of int
            The size of the shifts along the specified axes.
        axes : tuple of int
            The axes along which the shift is applied.
        input_shape : tuple of int
            Shape of the input tensor (nX, nY, nZ, nS, nTI, nTE).
        """
        nX, nY, nZ, nS, nTI, nTE = input_shape
        nW = torch.prod(torch.tensor(shift_number))

        self.shift_number = shift_number
        self.shift_size = shift_size
        self.axes = axes

        self.forward_shape = (1, nX, nY, nZ, nS, nTI, nTE)
        self.adjoint_shape = (nW, nX, nY, nZ, nS, nTI, nTE)

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        # Set the data type to float32
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the forward shift operation.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor on which the shift operation will be applied.

        Returns:
        -------
        torch.Tensor
            Output tensor after applying the forward shift.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = shift_forward(
            input_tensor, self.shift_number, self.shift_size, self.axes
        )
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the adjoint shift operation.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor on which the adjoint shift operation will be applied.

        Returns:
        -------
        torch.Tensor
            Output tensor after applying the adjoint shift.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = shift_adjoint(
            input_tensor, self.shift_number, self.shift_size, self.axes
        )
        return output_tensor.ravel().view(self.dtype)
