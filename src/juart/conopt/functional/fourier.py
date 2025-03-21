import warnings
from typing import Optional, Tuple, Union

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
    device: Optional[torch.DeviceObjType] = None,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) forward from uniform grid to
    non-uniform points.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (D, N) or (D, N, ...)
        with D dimensions (xyz) and N samples scaled between [-0.5, 0.5].
    x : torch.Tensor, Shape (C, R, P1, P2) or (C, R, P1, P2, ...)
        Input data on a uniform grid with C channels and R readout,
        P1 phase P2 partition samples.
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).
    device : str, optional
        Troch device.

    Returns
    -------
    torch.Tensor
        Nonuniform Fourier transform of `x` at the sampling points `k`
        with shape (C, N, ...).
    """

    if k.device != x.device:
        raise ValueError("k and x must be on the same device.")

    if device is not None:
        k = k.to(device)
        x = x.to(device)
    else:
        device = k.device

    # Ensure that the input kspace locations have correct scaling
    if torch.any(torch.abs(k) > 0.5):
        raise ValueError(
            "Non-uniform sampling points k must be scaled between -0.5 and 0.5."
        )

    # Ensure x will have at least shape (C, R, P1, P2, 1)
    if x.ndim < 4:
        raise ValueError(
            "Input data x must have at least three dimensions "
            "(channel, read, phase1, phase2)!"
        )
    elif x.ndim == 4:
        x = x.unsqueeze(4)  # Add additional dimension

    # Ensure k will have at least shape (D, N, 1)
    if k.ndim < 2:
        raise ValueError(
            "Non-uniform sampling points k must have at least two dimensions "
            "(dim, cols)."
        )
    elif k.ndim == 2:
        k = k.unsqueeze(2)

    # Get dimensions
    D, N, *add_axes_k = k.shape
    C, R, P1, P2, *add_axes_x = x.shape

    # Handle extra dimensions of k and x
    if add_axes_k != [1] and add_axes_k != add_axes_x:
        raise ValueError(
            "The additional axes in k and x must be the same "
            f"but are {add_axes_k} and {add_axes_x}."
        )

    # Ensure correct dimension handling
    if D == 1 and (P1 != 1 and P2 != 1):
        raise ValueError(
            f"For 1D input kspace locations P1 and P2 must be 1, but are {P1} and {P2}."
        )

    elif D == 2 and P2 != 1:
        raise ValueError(f"For 2D input kspace locations P2 must be 1, but is {P2}.")

    #
    norm = torch.sqrt(torch.tensor(R * P1 * P2, dtype=torch.float32, device=device))

    # Scale k to the correct range and convert to complex for finufft
    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)

    # Reshape x and k to have all additional axes as a single axis
    x = x.view(C, R, P1, P2, -1)
    k = k.view(D, N, -1)

    # Initialize output tensor
    y = torch.zeros((C, N, x.shape[-1]), dtype=x.dtype, device=device)

    for n_cha in range(C):
        for n_ax in range(x.shape[-1]):
            k_tmp = k[..., n_ax] if k.shape[-1] > 1 else k[..., 0]
            # Squeeze in case D=2 and any of R, P1, P2 is 1
            x_tmp = x[n_cha, ..., n_ax].squeeze()

            y[n_cha, ..., n_ax] = pytorch_finufft.functional.finufft_type2(
                points=k_tmp.contiguous(),
                targets=x_tmp.contiguous(),
                eps=eps,
            )

    y /= norm

    # Reshape output data to have all additional axes as in original shape
    if all(axis == 1 for axis in add_axes_x):
        y = y.view(C, N)
    else:
        y = y.view(C, N, *add_axes_x)

    return y


def nonuniform_fourier_transform_adjoint(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    eps: float = 1e-6,
    device: Optional[torch.DeviceObjType] = None,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) adjoint
    from non-uniform points to a uniform grid.

    Parameters
    ----------
    k : torch.Tensor, Shape (D, N) or (D, N, ...)
        Non-uniform sampling points with D dimensions (xyz) and N samples.
    x : torch.Tensor, Shape (N, ), (C, N) or (C, N, ...)
        Signal data of the non-uniform sampling points `k`.
        with C channels and N samples.
    n_modes : tuple of ints
        Shape of the grid to reconstruct to (e.g., (nx,), (nx, ny), (nx, ny, nz)).
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).
    device : torch.device, optional
        Device to run the computation.
        If None, the device of `k` will be used.

    Returns
    -------
    torch.Tensor, Shape (C, nx, ny, nz) or (C, nx, ny, nz, ...)
        Singal data on a uniform grid.
        If nx, ny or nz is not in n_modes, the corresponding output dimension will be 1.
    """
    if device is not None:
        k = k.to(device)
        x = x.to(device)
    else:
        device = k.device

        if x.device != device:
            warnings.warn(
                f"k and x are not on the same device. Copy x to {device}",
                stacklevel=1,
            )
            x = x.to(device)

    if torch.max(torch.abs(k)) > 0.5:
        raise ValueError(
            "Non-uniform sampling points k must be scaled between -0.5 and 0.5."
        )

    if k.shape[0] != len(n_modes):
        raise ValueError(
            "The number of dimensions in n_modes "
            "must be equal to the number of dimensions in k "
            f"but are {k.shape[0]} and {len(n_modes)}."
        )

    norm = torch.sqrt(torch.prod(torch.tensor(n_modes)))

    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)

    # Ensure x to have at least shape (C, N, 1)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # Add axis for channel dimension
    if x.ndim == 2:
        x = x.unsqueeze(2)  # Add axis for additional dimensions

    # Ensure k to have at least shape (D, N, 1)
    if k.ndim == 2:
        k = k.unsqueeze(2)  # Add axis for additional dimensions

    # Ensure that the input data has the correct shape
    C, N_x, *add_axes_x = x.shape
    D, N_k, *add_axes_k = k.shape

    if N_x != N_k:
        raise ValueError(
            "The number of columns in x and k must be the same "
            f"but are {N_x} and {N_k}."
        )
    N = N_x
    if add_axes_k != [1] and add_axes_k != add_axes_x:
        raise ValueError(
            "The additional axes in k and x must be the same "
            f"but are {add_axes_k} and {add_axes_x}."
        )
    add_axes = add_axes_x

    # Reshape input data from mulitple additional axes to a single additional axis
    x = x.reshape(C, N, -1)
    k = k.reshape(D, N, -1)

    # Create empty output tensor
    x_out = torch.zeros((C, *n_modes, x.shape[-1]), dtype=x.dtype, device=x.device)

    # Compute the adjoint NUFFT for each axis
    for n_cha in range(C):
        for n_ax in range(x.shape[-1]):
            k_tmp = k[..., 0] if k.shape[-1] == 1 else k[..., n_ax]
            x_tmp = x[n_cha, :, n_ax]
            x_out[n_cha, ..., n_ax] = pytorch_finufft.functional.finufft_type1(
                points=k_tmp.contiguous(),
                values=x_tmp.contiguous(),
                output_shape=n_modes,
                eps=eps,
            )

    x_out /= norm

    # Reshape output data to have all additional axes as in original shape
    x_out = x_out.reshape(C, *n_modes, *add_axes)

    # Ensure x has always nx, ny, nz dimensions
    if len(n_modes) < 3:
        mode_diff = 3 - len(n_modes)
        for _ in range(mode_diff):
            x_out = x_out.unsqueeze(len(n_modes) + 1)

    # If additional axes are [1], remove them
    if add_axes == [1]:
        x_out = x_out.squeeze(-1)

    return x_out
