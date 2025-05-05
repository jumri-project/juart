from typing import Tuple

import torch

from ..functional.fourier import (
    fourier_transform_forward,
    nonuniform_fourier_transform_adjoint,
)


def nonuniform_transfer_function(
    k: torch.Tensor,
    data_shape: Tuple[int, ...],
    oversampling: Tuple[int, ...] = 2,
) -> torch.Tensor:
    """
    Compute the non-uniform transfer function using PyTorch.

    Parameters
    ----------
    k : torch.Tensor
        The k-space trajectory of shape (D, N, ...)
        with D dimensions (kx,ky,kz) and N columns.
    data_shape : tuple of ints
        The shape of the complex data tensor of format (1, R, P1, P2, ...).
    oversampling : int or tuple of ints, optional
        The oversampling factors along each axis
        (eg. (OS_R,), (OS_R, OS_P1), (OS_R, OS_P1, OS_P2)).
        If an integer is provided,
        it is used for all axes (R, P1, P2)>1 in `data_shape`.
        Default is 2.
    eps : float, optional
        The error tolerance for the NUFFT adjoint operation.

    Returns
    -------
    transfer_function : torch.Tensor
        The transfer function with
         oversampled data_shape (1, OS_R*R, OS_P1*P1, OS_P2*P2, ...).
    """

    _, nX, nY, nZ, *add_axes_x = data_shape
    nDim, nCol, *add_axes_k = k.shape

    excl_axes_x = add_axes_x[len(add_axes_k) :]
    n_modes = tuple([n for n in [nX, nY, nZ] if n > 1])

    if len(n_modes) != nDim:
        raise ValueError(
            "The number of spacial encoding dimensions (R, P1, P2)"
            " in data_shape must match the number of dimensions in k."
        )

    # Apply oversampling
    if isinstance(oversampling, int):
        oversampling = (oversampling,) * len(n_modes)
    elif len(oversampling) != len(n_modes):
        raise ValueError(
            "Oversampling must be an integer or a tuple with length equal to the "
            "number of spacial encoding dimensions (R, P1, P2) in data_shape."
        )
    n_modes = tuple([n * o for n, o in zip(n_modes, oversampling)])

    # Create normalized input array (PyTorch tensor)
    norm = 1 / torch.prod(torch.tensor(oversampling, dtype=torch.float32))
    x = torch.ones((1, nCol, *add_axes_k), dtype=torch.complex64, device=k.device)
    x = x / norm

    # Compute the Point-Spread Function (PSF) using the non-uniform adjoint Fourier
    # transform for the dimensions where k changes
    PSF = nonuniform_fourier_transform_adjoint(k, x, n_modes)

    # Add the additioal dimensions of the output
    new_axes = PSF.shape + (1,) * len(excl_axes_x)
    new_shape = PSF.shape + tuple(excl_axes_x)
    PSF = PSF.reshape(new_axes).expand(new_shape)

    # Compute the transfer function using the forward Fourier transform
    fft_axes = tuple(range(1, len(n_modes) + 1))
    transfer_function = fourier_transform_forward(PSF, fft_axes)

    return transfer_function
