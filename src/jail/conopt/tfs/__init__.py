from typing import Tuple

import torch

from ..aux import crop_tensor, pad_tensor
from ..aux.fourier import fourier_transform_adjoint, fourier_transform_forward


def apply_transfer_function(
    x: torch.Tensor,
    transfer_function: torch.Tensor,
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Apply a transfer function in the Fourier domain using PyTorch.

    This function takes an input tensor `x`, transforms it into the Fourier domain,
    applies a given transfer function `transfer_function` in the Fourier space, and then
    transforms the result back to the original domain.

    Parameters:
    ----------
    x : torch.Tensor
        The input tensor to which the transfer function is applied.
    transfer_function : torch.Tensor
        The transfer function to apply in the Fourier domain.
    axes : tuple of ints
        The axes along which the Fourier transform is applied.

    Returns:
    -------
    torch.Tensor
        The tensor after applying the transfer function in the Fourier domain.
    """
    # Perform forward Fourier transform
    x = fourier_transform_forward(x, axes)

    # Apply the transfer function
    x = transfer_function * x

    # Perform inverse Fourier transform
    x = fourier_transform_adjoint(x, axes)

    return x.clone()  # Clone to ensure memory continuity


def apply_oversampled_transfer_function(
    x: torch.Tensor,
    transfer_function: torch.Tensor,
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Apply an oversampled transfer function in the Fourier domain using PyTorch.

    This function first pads the input tensor `x` to match the size of the transfer
    function `transfer_function` along the specified `axes`. It then applies the
    transfer function in the Fourier space, performs the inverse Fourier transform,
    and crops the result back to the original shape.

    Parameters:
    ----------
    x : torch.Tensor
        The input tensor to which the oversampled transfer function is applied.
    transfer_function : torch.Tensor
        The transfer function to apply in the Fourier domain.
    axes : tuple of ints
        The axes along which the Fourier transform is applied.

    Returns:
    -------
    torch.Tensor
        The tensor after applying the oversampled transfer function in the Fourier
        domain.
    """
    # Get the shape and pad it to match the transfer function's size along the specified
    # axes
    shape = x.shape
    pad_shape = list(shape)

    for ax in axes:
        pad_shape[ax] = transfer_function.shape[ax]  # Adjust the padding size

    # Pad the input tensor to match the size of transfer_function
    x = pad_tensor(x, pad_shape)

    # Perform forward Fourier transform
    x = fourier_transform_forward(x, axes)

    # Apply the transfer function
    x = transfer_function * x

    # Perform inverse Fourier transform
    x = fourier_transform_adjoint(x, axes)

    # Crop back to the original shape
    x = crop_tensor(x, shape)

    return x.clone()  # Clone to ensure memory continuity
