import warnings
from typing import Tuple

import finufft
import pytorch_finufft
import torch


def fourier_transform_forward(
    x: torch.Tensor,
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Compute the fast Fourier transform of an array using torch.fft on the GPU/CPU.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    axes : tuple of ints
        Axes over which to compute the FFT.

    Returns
    -------
    torch.Tensor
        FFT of the input tensor.
    """
    x = torch.fft.ifftshift(x, dim=axes)
    x = torch.fft.fftn(x, dim=axes, norm="ortho")
    x = torch.fft.fftshift(x, dim=axes)
    return x


def fourier_transform_adjoint(
    x: torch.Tensor,
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Compute the inverse fast Fourier transform of an array using torch.fft on the
    GPU/CPU.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    axes : tuple of ints
        Axes over which to compute the IFFT.

    Returns
    -------
    torch.Tensor
        IFFT of the input tensor.
    """
    x = torch.fft.ifftshift(x, dim=axes)
    x = torch.fft.ifftn(x, dim=axes, norm="ortho")
    x = torch.fft.fftshift(x, dim=axes)
    return x


def nufft2d_type1(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Tuple[int, ...] = None,
    eps: float = 1e-6,
    nthreads: int = 1,
) -> torch.Tensor:
    """
    Compute the NUFFT (type 1) converting non-uniform samples to a uniform grid.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (2, num_samples).
    x : torch.Tensor
        Input tensor of shape (num_samples, ...).
    n_modes : tuple of ints, optional
        Output grid size.
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).
    nthreads : int, optional
        Number of threads to use (default: 1).

    Returns
    -------
    torch.Tensor
        Fourier transformed data on a uniform grid.
    """
    y = finufft.nufft2d1(
        k[0, :].cpu().numpy(),
        k[1, :].cpu().numpy(),
        x.cpu().numpy(),
        n_modes=n_modes,
        eps=eps,
        nthreads=nthreads,
    )
    return torch.tensor(y, dtype=x.dtype, device=x.device)


def nufft2d_type2(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Tuple[int, ...] = None,
    eps: float = 1e-6,
    nthreads: int = 1,
) -> torch.Tensor:
    """
    Compute the NUFFT (type 2) converting uniform grid data to non-uniform points.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (2, num_samples).
    x : torch.Tensor
        Fourier domain data on a uniform grid.
    n_modes : tuple of ints, optional
        Size of the input grid.
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).
    nthreads : int, optional
        Number of threads to use (default: 1).

    Returns
    -------
    torch.Tensor
        The inverse NUFFT result at non-uniform points.
    """
    y = finufft.nufft2d2(
        k[0, :].cpu().numpy(),
        k[1, :].cpu().numpy(),
        x.cpu().numpy(),
        eps=eps,
        nthreads=nthreads,
    )
    return torch.tensor(y, dtype=x.dtype, device=x.device)


def nonuniform_fourier_transform_forward(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Tuple[int, ...],
    shape: Tuple[int, ...],
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) forward from non-uniform points to
    a uniform grid.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (2, num_samples).
    x : torch.Tensor
        Input data tensor.
    n_modes : tuple of ints
        Shape of the output grid (e.g., (nx, ny)).
    shape : tuple
        Shape of the input data.
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Fourier-transformed data on a uniform grid.
    """
    norm = torch.sqrt(torch.tensor(n_modes[0] * n_modes[1], dtype=torch.float32))

    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)
    y = torch.zeros(shape, dtype=x.dtype, device=x.device)

    for S, TI, TE in torch.cartesian_prod(
        torch.arange(shape[-3]), torch.arange(shape[-2]), torch.arange(shape[-1])
    ):
        y[:, 0, 0, :, S, TI, TE] = nufft2d_type2(
            k[:, :, S, TI, TE].clone(),
            x[:, :, :, 0, S, TI, TE].clone(),
            n_modes=n_modes[:2],
            eps=eps,
        )

    y /= norm
    return y


def nonuniform_fourier_transform_adjoint(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Tuple[int] | Tuple[int, int] | Tuple[int, int, int],
    eps: float = 1e-6,
    nthreads: int = 1,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) adjoint from a uniform grid to
    non-uniform points.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (D, N) or (D, N, ...) with D dimensions
        (xyz) and N samples.
    x : torch.Tensor
        Signal data of the non-uniform sampling points `k` of shape (N, ) or (C, N, ...)
        with C channels and N samples.
    n_modes : tuple of ints
        Shape of the grid to reconstruct to (e.g., (nx,), (nx, ny), (nx, ny, nz)).
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Singal data on a uniform grid with shape (*n_modes, C) or (*n_modes, C, ...).
    """
    if k.shape[0] > len(n_modes):
        warnings.warn(
            f"k is {k.shape[0]}D, but n_modes is {len(n_modes)}D. "
            "Adding additional dimensions to n_modes...",
            stacklevel=2,
        )

        n_modes = n_modes + (1,) * (k.shape[0] - len(n_modes))

    elif k.shape[0] < len(n_modes):
        warnings.warn(
            f"k is {k.shape[0]}D, but n_modes is {len(n_modes)}D. "
            "Adding additional dimensions with zeros to k ...",
            stacklevel=2,
        )

        _d_diff = len(n_modes) - k.shape[0]
        k = torch.cat(
            (k, torch.zeros((_d_diff, *k.shape[1:]), dtype=k.dtype, device=k.device)),
            dim=0,
        )

    norm = torch.sqrt(torch.prod(torch.tensor(n_modes)))

    k = 3 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)

    # Ensure x to have shape (C, N, ...)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # Add axis for channel dimension
    if x.ndim == 2:
        x = x.unsqueeze(2)  # Add axis for additional dimensions

    # Ensure k to have shape (D, N, ...)
    if k.ndim == 2:
        k = k.unsqueeze(2)  # Add axis for additional dimensions

    # Ensure that the input data has the correct shape
    num_cha, num_col_x, add_axes_x = x.shape[0], x.shape[1], x.shape[2:]
    num_dim, num_col_k, add_axes_k = k.shape[0], k.shape[1], k.shape[2:]

    if num_col_x != num_col_k:
        raise ValueError(
            "The number of columns in x and k must be the same "
            f"but are {num_col_x} and {num_col_k}."
        )
    num_col = num_col_x

    if add_axes_x != add_axes_k:
        raise ValueError(
            "The additional axes in x and k must be the same "
            f"but are {add_axes_x} and {add_axes_k}."
        )
    add_axes = add_axes_x

    # Reshape input data from mulitple additional axes to a single additional axis
    x = x.reshape(num_cha, num_col, -1)
    k = k.reshape(num_dim, num_col, -1)

    # Create empty outptu tensor
    x_out = torch.zeros(
        (*n_modes, num_cha, x.shape[-1]), dtype=x.dtype, device=x.device
    )

    # Compute the adjoint NUFFT for each axis
    for n_cha in range(num_cha):
        for n_ax in range(x.shape[-1]):
            k_tmp = k[..., n_ax].clone()
            x_tmp = x[n_cha, :, n_ax].clone()
            x_out[..., n_cha, n_ax] = pytorch_finufft.functional.finufft_type1(
                points=k_tmp,
                values=x_tmp,
                output_shape=n_modes,
                eps=eps,
                nthreads=nthreads,
            )

    x_out /= norm

    # Reshape output data to have all additional axes as in original shape
    x_out = x_out.reshape(*n_modes, num_cha, *add_axes)

    return x_out.squeeze()
