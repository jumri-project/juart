from typing import Tuple

import finufft
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
        k[0, :].numpy(),
        k[1, :].numpy(),
        x.numpy(),
        n_modes=n_modes,
        eps=eps,
        nthreads=nthreads,
    )
    return torch.tensor(y, dtype=torch.complex64)


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
        k[0, :].numpy(),
        k[1, :].numpy(),
        x.numpy(),
        eps=eps,
        nthreads=nthreads,
    )
    return torch.tensor(y, dtype=torch.complex64)


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
    y = torch.zeros(shape, dtype=torch.complex64)

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
    n_modes: Tuple[int, ...],
    shape: Tuple[int, ...],
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) adjoint from a uniform grid to
    non-uniform points.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (2, num_samples).
    x : torch.Tensor
        Fourier data on a uniform grid.
    n_modes : tuple of ints
        Shape of the input grid (e.g., (nx, ny)).
    shape : tuple
        Shape of the output data.
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Inverse Fourier-transformed data at non-uniform points.
    """
    norm = torch.sqrt(torch.tensor(n_modes[0] * n_modes[1], dtype=torch.float32))

    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)
    y = torch.zeros(shape, dtype=torch.complex64)

    for S, TI, TE in torch.cartesian_prod(
        torch.arange(shape[-3]), torch.arange(shape[-2]), torch.arange(shape[-1])
    ):
        y[:, :, :, 0, S, TI, TE] = nufft2d_type1(
            k[:, :, S, TI, TE].clone(),
            x[:, 0, 0, :, S, TI, TE].clone(),
            n_modes=n_modes[:2],
            eps=eps,
        )

    y /= norm
    return y
