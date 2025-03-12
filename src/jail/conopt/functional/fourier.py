import warnings
from typing import Tuple, Union

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
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) forward from uniform grid to
    non-uniform points.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (D, N) or (D, N, ...)
        with D dimensions (xyz) and N samples scaled between [-0.5, 0.5].
    x : torch.Tensor
        Input data on a uniform grid of shape (C, R, P, S) or (C, R, P, S, ...)
        with C channels and R readout, P phase and S slice samples.
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Fourier-transformed data on a uniform grid.
    """

    # Ensure that the input kspace locations have correct scaling
    if torch.any(k < -0.5) or torch.any(k > 0.5):
        raise ValueError(
            "Non-uniform sampling points k must be scaled between -0.5 and 0.5."
        )

    # Ensure x has at least shape (C, R, P, S, 1)
    if x.ndim < 4:
        raise ValueError("Input data x must have at least three dimensions.")
    elif x.ndim == 4:
        x = x.unsqueeze(4)  # Add additional dimension

    # Ensure k has at least shape (D, N, 1)
    if k.ndim < 2:
        raise ValueError(
            "Non-uniform sampling points k must have at least two dimensions."
        )
    elif k.ndim == 2:
        k = k.unsqueeze(2)

    D, N, *add_axes_k = k.shape
    C, R, P, S, *add_axes_x = x.shape

    if add_axes_k != add_axes_x:
        raise ValueError(
            "The additional axes in x and k must be the same "
            f"but are {add_axes_x} and {add_axes_k}."
        )

    # Ensure correct dimension handling
    if D == 1 and (R * P * S != max(R, P, S)):
        raise ValueError(
            "For 1D input kspace locations,"
            " two of the data dimensions (R,P,S) must be 1."
        )
    if D == 2 and all(dim != 1 for dim in [R, P, S]):
        raise ValueError(
            "For 2D input kspace locations,"
            " one of the data dimensions (R,P,S) must be 1."
        )

    norm = torch.sqrt(torch.tensor(R * P * S, dtype=torch.float32))

    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)

    # Reshape x and k to have all additional axes as a single axis
    x = x.reshape(C, R, P, S, -1)
    k = k.reshape(D, N, -1)

    y = torch.zeros((C, D, N, x.shape[-1]), dtype=x.dtype, device=x.device)
    for n_cha in range(C):
        for n_ax in range(x.shape[-1]):
            k_tmp = k[..., n_ax]
            # Squeeze in case D=2 and any of R, P, S is 1
            x_tmp = x[n_cha, ..., n_ax].squeeze()

            y_tmp = pytorch_finufft.functional.finufft_type2(
                points=k_tmp,
                targets=x_tmp,
                eps=eps,
            )

            y[n_cha, ..., n_ax] = y_tmp

    y /= norm

    # Reshape output data to have all additional axes as in original shape
    y = y.reshape(C, D, N, *add_axes_x)

    return y


def nonuniform_fourier_transform_adjoint(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    eps: float = 1e-6,
    nthreads: int = 1,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) adjoint
    from non-uniform points to a uniform grid.

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

    # Create empty output tensor
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

    # Remove singleton dimensions, ensuring the result has at least three dimensions
    new_shape = x_out.shape[:3] + tuple(dim for dim in x_out.shape[3:] if dim != 1)
    x_out = x_out.reshape(new_shape)

    return x_out
