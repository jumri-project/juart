from typing import Tuple

import torch

from ..aux.fourier import (
    fourier_transform_forward,
    nonuniform_fourier_transform_adjoint,
)


def nonuniform_transfer_function(
    k: torch.Tensor,
    shape: Tuple[int, ...],
    oversampling: int = 2,
    eps: int = 1e-6,
) -> torch.Tensor:
    """
    Compute the non-uniform transfer function using PyTorch.

    Parameters
    ----------
    k : torch.Tensor
        The k-space trajectory.
    shape : tuple of ints
        The shape of the data.
    oversampling : int or tuple of ints, optional
        The oversampling factors along each axis.
    eps : float, optional
        The error tolerance for the NUFFT adjoint operation.

    Returns
    -------
    transfer_function : torch.Tensor
        The transfer function.
    """

    nX, nY, nZ, nS, nTI, nTE, nK = shape

    if k.shape[0] == 2:
        axis = (1, 2)
        nXs = oversampling[0] * nX
        nYs = oversampling[1] * nY
        nZs = nZ

    elif k.shape[0] == 3:
        axis = (1, 2, 3)
        nXs = oversampling[0] * nX
        nYs = oversampling[1] * nY
        nZs = oversampling[2] * nZ

    norm = 1 / torch.prod(torch.tensor(oversampling, dtype=torch.float32))

    # Create normalized input array (PyTorch tensor)
    x = torch.ones((1, 1, 1, nK, nS, nTI, nTE), dtype=torch.complex64) / norm

    # Compute the Point-Spread Function (PSF) using the non-uniform adjoint Fourier
    # transform
    PSF = nonuniform_fourier_transform_adjoint(
        k, x, (nXs, nYs, nZs), (1, nXs, nYs, nZs, nS, nTI, nTE), eps=eps
    )

    # Compute the transfer function (transfer_function) using the forward Fourier
    # transform
    transfer_function = fourier_transform_forward(PSF, axis)

    return transfer_function
