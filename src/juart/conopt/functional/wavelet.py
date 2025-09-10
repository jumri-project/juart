from typing import List, Optional, Tuple

import torch

from .fourier import fourier_transform_adjoint, fourier_transform_forward


def wavelet_filters(
    wavelet: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the scaling filter h and wavelet filter g for the specified wavelet.

    Parameters:
    ----------
    wavelet : str
        The type of wavelet ('db1', 'db2', 'db3', 'db4').
    dtype : torch.dtype, optional
        Data type for the filters (default: torch.float32).

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        h : Scaling filter coefficients.
        g : Wavelet filter coefficients.
    """
    if wavelet == "db1":
        h = torch.tensor([1, 1], dtype=dtype, device=device)
    elif wavelet == "db2":
        h = torch.tensor(
            [0.6830127, 1.1830127, 0.3169873, -0.1830127], dtype=dtype, device=device
        )
    elif wavelet == "db3":
        h = torch.tensor(
            [0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175],
            dtype=dtype,
            device=device,
        )
    elif wavelet == "db4":
        h = torch.tensor(
            [
                0.32580343,
                1.01094572,
                0.89220014,
                -0.03957503,
                -0.26450717,
                0.0436163,
                0.0465036,
                -0.01498699,
            ],
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported wavelet type: {wavelet}")

    # Compute g (wavelet filter)
    g = torch.flip(h, dims=[0]) * (-1) ** torch.arange(
        h.size(0), dtype=dtype, device=device
    )

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))

    return h / sqrt2, g / sqrt2


def wavelet_transfer_functions(
    wavelet: str,
    levels: int,
    length: int,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the wavelet transfer functions Hc and Gc for the given wavelet, levels, and
    signal length.

    Parameters:
    ----------
    wavelet : str
        The type of wavelet ('db1', 'db2', etc.).
    levels : int
        Number of decomposition levels.
    length : int
        Length of the signal.
    dtype : torch.dtype, optional
        Data type for the transfer functions (default: torch.complex64).

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Hc : Scaling transfer functions of shape (length, levels).
        Gc : Wavelet transfer functions of shape (length, levels).
    """
    h, g = wavelet_filters(wavelet, device=device)

    H = torch.zeros((levels, length), dtype=dtype, device=device)
    G = torch.zeros((levels, length), dtype=dtype, device=device)

    indices = h.size(0)

    for level in range(levels):
        for index in range(indices):
            idx = (1 - index) * (2**level) % length
            H[level, idx] = h[index]
            G[level, idx] = g[index]

    H = torch.fft.fftshift(H, dim=1)
    G = torch.fft.fftshift(G, dim=1)

    sqrt_length = torch.sqrt(torch.tensor(length, dtype=dtype, device=device))
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))

    H = fourier_transform_forward(H, axes=(1,)) * sqrt_length / sqrt2
    G = fourier_transform_forward(G, axes=(1,)) * sqrt_length / sqrt2

    Hc = torch.zeros((levels, length), dtype=dtype, device=device)
    Gc = torch.zeros((levels, length), dtype=dtype, device=device)

    for level in range(levels):
        if level == 0:
            prod_H = torch.ones(length, dtype=dtype, device=device)
        else:
            prod_H = torch.prod(H[:level, :], dim=0)
        Hc[level, :] = prod_H * H[level, :]
        Gc[level, :] = prod_H * G[level, :]

    return Hc.T, Gc.T


def wavelet_transfer_functions_nd(
    H: List[torch.Tensor],
    G: List[torch.Tensor],
    shape: Tuple[int, ...],
    axes: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Constructs the wavelet transfer functions F for multidimensional data.

    Parameters:
    ----------
    H : list of torch.Tensor
        List of scaling transfer functions for each axis.
    G : list of torch.Tensor
        List of wavelet transfer functions for each axis.
    shape : Tuple[int, ...]
        Shape of the data (nL, nX, nY, nZ, nS, nTI, nTE).
    axes : Tuple[int, ...]
        Axes along which to apply the wavelet transform.

    Returns:
    -------
    List[torch.Tensor]
        List of transfer functions for each axis.
    """
    nL, *add_axes = shape

    if len(axes) == 2:
        Hx, Hy = H[0], H[1]
        Gx, Gy = G[0], G[1]

        Fx = torch.cat((Gx[:, :, None], Hx[:, :, None], Gx[:, :, None]), dim=2)
        Fx = Fx.reshape(-1, nL - 1)
        Fx = torch.cat((Fx, Hx[:, -1:]), dim=1)
        Fx = Fx.reshape(Fx.shape[:1] + (len(add_axes) - 1) * (1,) + Fx.shape[1:])
        Fx = torch.movedim(Fx, 0, axes[0])

        Fy = torch.cat((Hy[:, :, None], Gy[:, :, None], Gy[:, :, None]), dim=2)
        Fy = Fy.reshape(-1, nL - 1)
        Fy = torch.cat((Fy, Hy[:, -1:]), dim=1)
        Fy = Fy.reshape(Fy.shape[:1] + (len(add_axes) - 1) * (1,) + Fy.shape[1:])
        Fy = torch.movedim(Fy, 0, axes[1])

        F = [Fx.to(device), Fy.to(device)]

    elif len(axes) == 3:
        Hx, Hy, Hz = H[0], H[1], H[2]
        Gx, Gy, Gz = G[0], G[1], G[2]

        Fx = torch.cat((Gx[:, :, None], Hx[:, :, None], Gx[:, :, None]), dim=2)
        Fx = Fx.reshape(-1, nL - 1)
        Fx = torch.cat((Fx, Hx[:, -1:]), dim=1)
        Fx = Fx.reshape(Fx.shape[:1] + (len(add_axes) - 1) * (1,) + Fx.shape[1:])
        Fx = torch.movedim(Fx, 0, axes[0])

        Fy = torch.cat((Hy[:, :, None], Gy[:, :, None], Gy[:, :, None]), dim=2)
        Fy = Fy.reshape(-1, nL - 1)
        Fy = torch.cat((Fy, Hy[:, -1:]), dim=1)
        Fy = Fy.reshape(Fy.shape[:1] + (len(add_axes) - 1) * (1,) + Fy.shape[1:])
        Fy = torch.movedim(Fy, 0, axes[1])

        Fz = torch.cat((Hz[:, :, None], Gz[:, :, None], Gz[:, :, None]), dim=2)
        Fz = Fz.reshape(-1, nL - 1)
        Fz = torch.cat((Fz, Hz[:, -1:]), dim=1)
        Fz = Fz.reshape(Fz.shape[:1] + (len(add_axes) - 1) * (1,) + Fz.shape[1:])
        Fz = torch.movedim(Fz, 0, axes[2])

        F = [Fx.to(device), Fy.to(device), Fz.to(device)]

    else:
        raise ValueError(
            "Unsupported number of axes. Only 2D and 3D wavelets are supported."
        )

    return F


def wavelet_transform_forward(
    input_tensor: torch.Tensor, F: List[torch.Tensor], axes: Tuple[int, ...]
) -> torch.Tensor:
    """
    Perform the forward wavelet transform on the input tensor.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        Input tensor to be transformed.
    F : list of torch.Tensor
        List of transfer functions for each axis.
    axes : Tuple[int, ...]
        Axes along which to apply the wavelet transform.

    Returns:
    -------
    torch.Tensor
        Wavelet coefficients with levels as the first dimension.
    """
    if len(axes) == 2:
        x_fft = fourier_transform_forward(input_tensor, axes=axes)
        y = x_fft[..., None] * F[0] * F[1]
        y = fourier_transform_adjoint(y, axes=axes)

    elif len(axes) == 3:
        x_fft = fourier_transform_forward(input_tensor, axes=axes)
        y = x_fft[..., None] * F[0] * F[1] * F[2]
        y = fourier_transform_adjoint(y, axes=axes)

    else:
        raise ValueError(
            "Unsupported number of axes. Only 2D and 3D wavelets are supported."
        )

    # Move the levels dimension to the first dimension
    y = torch.moveaxis(y, -1, 0)

    y = y.contiguous()

    return y


def wavelet_transform_adjoint(
    input_tensor: torch.Tensor, F: List[torch.Tensor], axes: Tuple[int, ...]
) -> torch.Tensor:
    """
    Perform the adjoint wavelet transform on the input tensor.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        Wavelet coefficients with levels as the first dimension.
    F : list of torch.Tensor
        List of transfer functions for each axis.
    axes : Tuple[int, ...]
        Axes along which to apply the wavelet transform.

    Returns:
    -------
    torch.Tensor
        Reconstructed tensor after the adjoint wavelet transform.
    """
    # Move the levels dimension back to the last dimension
    input_tensor = torch.moveaxis(input_tensor, 0, -1)

    if len(axes) == 2:
        x_fft = fourier_transform_forward(input_tensor, axes=axes)
        y = x_fft * torch.conj(F[0]) * torch.conj(F[1])
        y = torch.sum(y, dim=-1)
        y = fourier_transform_adjoint(y, axes=axes)

    elif len(axes) == 3:
        x_fft = fourier_transform_forward(input_tensor, axes=axes)
        y = x_fft * torch.conj(F[0]) * torch.conj(F[1]) * torch.conj(F[2])
        y = torch.sum(y, dim=-1)
        y = fourier_transform_adjoint(y, axes=axes)

    else:
        raise ValueError(
            "Unsupported number of axes. Only 2D and 3D wavelets are supported."
        )

    y = y.contiguous()

    return y
