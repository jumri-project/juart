from typing import Tuple

import torch

from ..functional import crop_tensor, pad_tensor
from ..functional.fourier import fourier_transform_adjoint, fourier_transform_forward


def apply_transfer_function(
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
    if pad_shape != shape:
        x = pad_tensor(x, pad_shape)

    # Perform forward Fourier transform
    x = fourier_transform_forward(x, axes)

    # Apply the transfer function
    x = transfer_function * x

    # Perform inverse Fourier transform
    x = fourier_transform_adjoint(x, axes)

    # Crop back to the original shape
    if pad_shape != shape:
        x = crop_tensor(x, shape)

    return x.clone()  # Clone to ensure memory continuity
