from typing import Tuple

import torch

from ..aux.fourier import (
    fourier_transform_adjoint,
    fourier_transform_forward,
    nonuniform_fourier_transform_adjoint,
    nonuniform_fourier_transform_forward,
)
from ..tfs import apply_oversampled_transfer_function
from ..tfs.fourier import nonuniform_transfer_function
from . import LinearOperator


class FourierTransformOperator(LinearOperator):
    """
    Applies the standard Fourier transform along specified axes.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        axes: Tuple[int, ...],
    ):
        """
        Initialize the FourierTransformOperator.

        Parameters:
        ----------
        input_shape : tuple of int
            Shape of the input tensor.
        axes : tuple of int
            Axes along which to perform the Fourier transform.
        """
        self.shape = (
            2 * torch.prod(torch.tensor(input_shape)),
            2 * torch.prod(torch.tensor(input_shape)),
        )
        self.forward_shape = input_shape
        self.adjoint_shape = input_shape
        self.axes = axes

        # Set the data type to float32 and internal dtype to complex64
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward Fourier transform.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to be transformed.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the Fourier transform.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = fourier_transform_forward(input_tensor, self.axes)
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adjoint (inverse) Fourier transform.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to apply the adjoint (inverse) Fourier transform.

        Returns:
        -------
        torch.Tensor
            The result after applying the inverse Fourier transform.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = fourier_transform_adjoint(input_tensor, self.axes)
        return output_tensor.ravel().view(self.dtype)


class NonuniformFourierTransformOperator(LinearOperator):
    """
    Applies a non-uniform Fourier transform based on k-space coordinates.
    """

    def __init__(
        self,
        k: torch.Tensor,
        input_shape: Tuple[int, ...],
    ):
        """
        Initialize the NonuniformFourierTransformOperator.

        Parameters:
        ----------
        k : torch.Tensor
            K-space coordinates for the non-uniform Fourier transform.
        input_shape : tuple of int
            Shape of the input tensor.
        """
        nC, nX, nY, nZ, nS, nTI, nTE = input_shape
        nK = k.shape[1]

        self.k = k
        self.forward_shape = (nC, nX, nY, nZ, nS, nTI, nTE)
        self.adjoint_shape = (nC, 1, 1, nK, nS, nTI, nTE)
        self.n_modes = (nX, nY, nZ)

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        # Set the data type to float32 and internal dtype to complex64
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward non-uniform Fourier transform.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to be transformed.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the non-uniform Fourier transform.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )
        output_tensor = nonuniform_fourier_transform_forward(
            self.k, input_tensor, self.n_modes, self.adjoint_shape
        )
        return output_tensor.ravel().view(self.dtype)

    def _rmatvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adjoint (inverse) non-uniform Fourier transform.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to apply the adjoint (inverse) non-uniform Fourier transform.

        Returns:
        -------
        torch.Tensor
            The result after applying the inverse non-uniform Fourier transform.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = nonuniform_fourier_transform_adjoint(
            self.k, input_tensor, self.n_modes, self.forward_shape
        )
        return output_tensor.ravel().view(self.dtype)


class NonuniformFourierTransformNormalOperator(LinearOperator):
    """
    Applies a normalized non-uniform Fourier transform with oversampling.
    """

    def __init__(
        self,
        k: torch.Tensor,
        input_shape: Tuple[int, ...],
        oversampling: Tuple[int, ...] = (2, 2),
        nonuniform_axes: Tuple[int, ...] = (1, 2),
    ):
        """
        Initialize the NonuniformFourierTransformNormalOperator.

        Parameters:
        ----------
        k : torch.Tensor
            K-space coordinates for the non-uniform Fourier transform.
        input_shape : tuple of int
            Shape of the input tensor.
        oversampling : tuple of int, optional
            Oversampling factor (default is (2, 2)).
        nonuniform_axes : tuple of int, optional
            Axes along which the non-uniform Fourier transform is applied
            (default is (1, 2)).
        """
        nC, nX, nY, nZ, nS, nTI, nTE = input_shape
        nK = k.shape[1]

        self.k = k
        self.forward_shape = (nC, nX, nY, nZ, nS, nTI, nTE)
        self.adjoint_shape = (nC, nX, nY, nZ, nS, nTI, nTE)

        # Compute the non-uniform transfer function
        self.transfer_function = nonuniform_transfer_function(
            k,
            (nX, nY, nZ, nS, nTI, nTE, nK),
            oversampling=oversampling,
        )
        self.nonuniform_axes = nonuniform_axes

        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        # Set the data type to float32 and internal dtype to complex64
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward normalized non-uniform Fourier transform.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to be transformed.

        Returns:
        -------
        torch.Tensor
            Transformed tensor after applying the normalized non-uniform Fourier
            transform.
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
        Perform the adjoint normalized non-uniform Fourier transform.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input tensor to apply the adjoint (inverse) normalized non-uniform Fourier
            transform.

        Returns:
        -------
        torch.Tensor
            The result after applying the adjoint normalized non-uniform Fourier
            transform.
        """
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.adjoint_shape
        )
        output_tensor = apply_oversampled_transfer_function(
            input_tensor, self.transfer_function, self.nonuniform_axes
        )
        return output_tensor.ravel().view(self.dtype)
