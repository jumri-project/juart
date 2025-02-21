import itertools
from typing import Tuple

import torch


def system_matrix_shape(
    input_shape: Tuple[int, int, int],
    window_shape: Tuple[int, int, int],
) -> Tuple[int, int]:
    """
    Compute the shape parameters nH1 and nH2 for the system matrix.

    Parameters:
    ----------
    input_shape : Tuple[int, int, int]
        The shape of the input tensor (nX, nY, nZ).
    window_shape : Tuple[int, int, int]
        The window sizes (wX, wY, wZ) for each dimension.

    Returns:
    -------
    Tuple[int, int]
        nH1: Product of window dimensions.
        nH2: Product of shifted dimensions (sX, sY, sZ).
    """
    nX, nY, nZ = input_shape
    wX, wY, wZ = window_shape

    assert nX >= wX, "nX must be >= wX"
    assert nY >= wY, "nY must be >= wY"
    assert nZ >= wZ, "nZ must be >= wZ"

    sX = nX - wX + 1
    sY = nY - wY + 1
    sZ = nZ - wZ + 1

    nH1 = wX * wY * wZ
    nH2 = sX * sY * sZ

    return nH1, nH2


def system_matrix_forward(
    input_tensor: torch.Tensor,
    input_shape: Tuple[int, ...],
    window_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Perform the forward operation of the system matrix.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        Input tensor of shape (nC, nX, nY, nZ, nS, nTI, nTE).
    input_shape : Tuple[int, ...]
        Shape of the input tensor.
    window_shape : Tuple[int, int, int]
        Window sizes (wX, wY, wZ) for each dimension.

    Returns:
    -------
    torch.Tensor
        Output tensor after applying the system matrix forward operation.
    """
    nC, nX, nY, nZ, nS, nTI, nTE = input_shape
    wX, wY, wZ = window_shape

    assert nX >= wX, "nX must be >= wX"
    assert nY >= wY, "nY must be >= wY"
    assert nZ >= wZ, "nZ must be >= wZ"

    sX = nX - wX + 1
    sY = nY - wY + 1
    sZ = nZ - wZ + 1

    device = input_tensor.device
    dtype = input_tensor.dtype

    # Initialize output tensor
    output_tensor = torch.zeros(
        (nC, wX * wY * wZ, sX * sY * sZ, nS, nTI, nTE), dtype=dtype, device=device
    )

    # Iterate over window positions
    for iX, iY, iZ in itertools.product(range(wX), range(wY), range(wZ)):
        idx = iZ + iY * wZ + iX * wY * wZ
        x_slice = input_tensor[:, iX : iX + sX, iY : iY + sY, iZ : iZ + sZ, ...].clone()
        x_slice = x_slice.reshape(nC, sX * sY * sZ, nS, nTI, nTE)
        output_tensor[:, idx, ...] = x_slice

    # Normalize the output
    output_tensor = output_tensor / torch.sqrt(
        torch.tensor(wX * wY * wZ, dtype=dtype, device=device)
    )
    return output_tensor


def system_matrix_adjoint(
    input_tensor: torch.Tensor,
    input_shape: Tuple[int, ...],
    window_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Perform the adjoint operation of the system matrix.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        Input tensor of shape (nC, wX * wY * wZ, sX * sY * sZ, nS, nTI, nTE).
    input_shape : Tuple[int, ...]
        Shape of the output tensor.
    window_shape : Tuple[int, int, int]
        Window sizes (wX, wY, wZ) for each dimension.

    Returns:
    -------
    torch.Tensor
        Output tensor after applying the system matrix adjoint operation.
    """
    nC, nX, nY, nZ, nS, nTI, nTE = input_shape
    wX, wY, wZ = window_shape

    assert nX >= wX, "nX must be >= wX"
    assert nY >= wY, "nY must be >= wY"
    assert nZ >= wZ, "nZ must be >= wZ"

    sX = nX - wX + 1
    sY = nY - wY + 1
    sZ = nZ - wZ + 1

    device = input_tensor.device
    dtype = input_tensor.dtype

    # Initialize output tensor
    output_tensor = torch.zeros(
        (nC, nX, nY, nZ, nS, nTI, nTE), dtype=dtype, device=device
    )

    # Iterate over window positions
    for iX, iY, iZ in itertools.product(range(wX), range(wY), range(wZ)):
        idx = iZ + iY * wZ + iX * wY * wZ
        x_slice = input_tensor[:, idx, ...].clone()
        x_slice = x_slice.reshape(nC, sX, sY, sZ, nS, nTI, nTE)
        output_tensor[:, iX : iX + sX, iY : iY + sY, iZ : iZ + sZ, ...] += x_slice

    # Normalize the output
    output_tensor = output_tensor / torch.sqrt(
        torch.tensor(wX * wY * wZ, dtype=dtype, device=device)
    )
    return output_tensor


def system_matrix_normal(
    input_tensor: torch.Tensor,
    input_shape: Tuple[int, ...],
    window_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Compute the normal operator (A^T A) of the system matrix on the input tensor.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        Input tensor of shape (nC, nX, nY, nZ, nS, nTI, nTE).
    input_shape : Tuple[int, ...]
        Shape of the input tensor.
    window_shape : Tuple[int, int, int]
        Window sizes (wX, wY, wZ) for each dimension.

    Returns:
    -------
    torch.Tensor
        Output tensor after applying the normal operator.
    """
    nC, nX, nY, nZ, nS, nTI, nTE = input_shape
    wX, wY, wZ = window_shape

    assert nX >= wX, "nX must be >= wX"
    assert nY >= wY, "nY must be >= wY"
    assert nZ >= wZ, "nZ must be >= wZ"

    sX = nX - wX + 1
    sY = nY - wY + 1
    sZ = nZ - wZ + 1

    device = input_tensor.device
    dtype = input_tensor.dtype

    # Initialize output tensor
    output_tensor = torch.zeros(input_tensor.shape, dtype=dtype, device=device)

    # Iterate over window positions
    for iX, iY, iZ in itertools.product(range(wX), range(wY), range(wZ)):
        output_tensor[:, iX : iX + sX, iY : iY + sY, iZ : iZ + sZ, ...] += input_tensor[
            :, iX : iX + sX, iY : iY + sY, iZ : iZ + sZ, ...
        ]

    # Normalize the output
    output_tensor = output_tensor / (wX * wY * wZ)
    return output_tensor
