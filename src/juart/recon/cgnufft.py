import warnings
from collections.abc import Sequence
from typing import Optional, Union

import torch

from ..conopt.linops.identity import IdentityOperator
from ..conopt.linops.tf import TransferFunctionOperator
from ..conopt.proxops.linear import conjugate_gradient
from ..conopt.tfs.fourier import (
    nonuniform_fourier_transform_adjoint,
    nonuniform_transfer_function,
)


def cgnufft(
    ksp: torch.Tensor,
    ktraj: torch.Tensor,
    img_size: Union[Sequence[int], int],
    maxiter: int = 10,
    l2_reg: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: int = 0,
) -> torch.Tensor:
    """
    Perform Conjugate Gradient Non-Uniform Fast Fourier Transform (CGNUFFT).

    Parameters
    ----------
    ksp : torch.Tensor, shape (C, N), complex
        The complex k-space data with C channels and N samples.
    ktraj : torch.Tensor, shape (D, N)
        The trajectory in k-space with D spatial dimensions and N samples.
        Scaled between -0.5 and 0.5.
    maxiter : int, optional
        Maximum number of iterations for the conjugate gradient solver (default is 10).
    l2_reg : float, optional
        L2 regularization parameter (default is 0.0).
    device : Optional[torch.device], optional
        Device to perform computations on
        (default is None, which uses the current device).

    Returns
    -------
    torch.Tensor
        The reconstructed image from the k-space data.

    NOTE: This function is under development and may not be fully functional yet.
    """

    if device is None:
        device = ksp.device

    if ksp.shape[1] != ktraj.shape[1]:
        raise ValueError("The number of samples in k-space and trajectory must match.")

    if ksp.ndim != 2 or ktraj.ndim != 2:
        raise ValueError(
            "k-space and trajectory must be 2D tensors."
            "Higher dimensional data is not supported yet."
        )

    ktraj_min, ktraj_max = ktraj.min(), ktraj.max()
    if ktraj_min < -1 or ktraj_max > 1:
        warnings.warn(
            "Trajectory values should be scaled between -0.5 and 0.5 "
            f"but are {ktraj_min:.3f} and {ktraj_max:.3f}. "
            "This may lead to unexpected results.",
            stacklevel=2,
        )

    num_cha, num_col = ksp.shape
    num_dim = ktraj.shape[0]

    if num_dim not in (2, 3):
        raise ValueError(
            f"Only 2D and 3D trajectories are supported. Received {num_dim} dimensions."
        )

    # Get correct image size
    if isinstance(img_size, int):
        if num_dim == 2:
            img_size = (img_size, img_size, 1)
        else:
            img_size = (img_size, img_size, img_size)

    elif isinstance(img_size, Sequence):
        if len([d for d in img_size if d > 1]) != num_dim:
            raise ValueError(
                "Number of nonsingleton dimensions in image size "
                "must match the number of dimensions in the trajectory. "
                f"Expected {num_dim} dimensions, got {len(img_size)}."
            )
        if num_dim == 2:
            if len(img_size) == 2:
                img_size = (*img_size, 1)
            elif len(img_size) == 1:
                img_size = (img_size[0], img_size[0], 1)
        else:
            img_size = tuple(img_size)

    regridded_data = nonuniform_fourier_transform_adjoint(
        k=ktraj,
        x=ksp,
        n_modes=tuple([d for d in img_size if d > 1]),
    )

    transfer_function = nonuniform_transfer_function(
        k=ktraj, data_shape=(1, *regridded_data.shape[1:])
    )

    transfer_function_operator = TransferFunctionOperator(
        transfer_function, regridded_data.shape, device=device
    )

    Ident = IdentityOperator(regridded_data.shape, device=device)

    A = transfer_function_operator + l2_reg * Ident

    b = regridded_data.view(torch.float32).ravel()

    img, residual = conjugate_gradient(
        A,
        b,
        x=torch.zeros_like(b),
        residual=[],
        maxiter=maxiter,
        progressbar=verbose > 0,
    )

    img = img.view(torch.complex64).reshape(num_cha, *img_size)

    return img
