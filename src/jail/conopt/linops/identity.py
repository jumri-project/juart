from typing import Tuple

import torch

from . import LinearOperator


class IdentityOperator(LinearOperator):
    """
    Identity operator for linear operations, returning the input tensor unchanged.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
    ):
        """
        Initialize the IdentityOperator.

        Parameters:
        ----------
        input_shape : tuple of int
            Shape of the input tensor.
        """
        # Define the shape of the identity operator as (2 * product of input shape)
        self.shape = (
            2 * torch.prod(torch.tensor(input_shape)),
            2 * torch.prod(torch.tensor(input_shape)),
        )

        self.shape = (int(self.shape[0]), int(self.shape[1]))

        # Set the data types
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward identity operation (return input unchanged).

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to be returned unchanged.

        Returns:
        -------
        torch.Tensor
            The same input tensor.
        """
        return input_tensor

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adjoint identity operation (return input unchanged).

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to be returned unchanged.

        Returns:
        -------
        torch.Tensor
            The same input tensor.
        """
        return input_tensor
