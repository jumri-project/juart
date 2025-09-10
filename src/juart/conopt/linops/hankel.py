from typing import Optional, Tuple

import torch

from ..functional.hankel import (
    block_hankel_adjoint,
    block_hankel_forward,
    block_hankel_normal,
    block_hankel_shape,
)
from . import LinearOperator


class BlockHankelOperator(LinearOperator):
    """
    Implements a Block Hankel operator, which performs forward and adjoint operations
    for structured low-rank matrix approximation.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        normalize: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the BlockHankelOperator.

        Parameters:
        ----------
        input_shape : tuple of int
            Shape of the input tensor (nX, nY, nZ, nS, nT1, nT2).
        normalize : bool, optional
            Whether to normalize the operator (default is False).
        """
        *add_axes, nT1, nT2 = input_shape
        M, N, P, Q = block_hankel_shape((nT1, nT2))

        self.forward_shape = (*add_axes, nT1, nT2)
        self.adjoint_shape = (*add_axes, M * P, N * Q)

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        self.normalize = normalize

        # Set the data type to float32
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64
        self.device = device

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward block Hankel operation.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to be transformed.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the forward block Hankel operation.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = block_hankel_forward(
            input_tensor, self.forward_shape, self.normalize
        )
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adjoint block Hankel operation.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to apply the adjoint block Hankel operation.

        Returns:
        -------
        torch.Tensor
            The result after applying the adjoint block Hankel operation.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = block_hankel_adjoint(
            input_tensor, self.forward_shape, self.normalize
        )
        return output_tensor.ravel().view(self.dtype)


class BlockHankelNormalOperator(LinearOperator):
    """
    Implements a normalized Block Hankel operator with diagonal normalization.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        normalize: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the BlockHankelNormalOperator.

        Parameters:
        ----------
        input_shape : tuple of int
            Shape of the input tensor (nX, nY, nZ, nS, nT1, nT2).
        normalize : bool, optional
            Whether to normalize the operator (default is False).
        """
        *add_axes, nT1, nT2 = input_shape
        M, N, P, Q = block_hankel_shape((nT1, nT2))

        # Set the data type to float32
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64
        self.device = device

        self.forward_shape = (*add_axes, nT1, nT2)
        self.adjoint_shape = (*add_axes, nT1, nT2)

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        # Create diagonal normalization using block Hankel normal operator
        self.diagonal_normalization = block_hankel_normal(
            torch.ones(input_shape, dtype=self.internal_dtype, device=self.device),
            input_shape,
            normalize,
        )

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward operation with the diagonal normalization.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to be transformed.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the diagonal normalization.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = input_tensor * self.diagonal_normalization
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adjoint operation with the diagonal normalization.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to apply the adjoint operation.

        Returns:
        -------
        torch.Tensor
            The result after applying the adjoint operation with diagonal normalization.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = input_tensor * self.diagonal_normalization
        return output_tensor.ravel().view(self.dtype)
