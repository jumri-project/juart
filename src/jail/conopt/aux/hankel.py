from typing import Tuple

import torch
from torch import jit


@jit.script
def block_hankel_shape(
    shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Compute the Hankel block shape from the input shape.

    Parameters:
    ----------
    shape : Tuple[int, int]
        The input tensor shape.

    Returns:
    -------
    Tuple[int, int, int, int]
        M, N, P, Q values representing Hankel block dimensions.
    """
    M = shape[0] // 2 + 1
    N = shape[0] - M + 1

    P = shape[1] // 2 + 1
    Q = shape[1] - P + 1

    return M, N, P, Q


@jit.script
def block_hankel_forward_kernel(
    input_tensor: torch.Tensor,
    M: int,
    N: int,
    P: int,
    Q: int,
    output_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Forward operation for the block Hankel matrix (in-place modification of
    output_tensor).

    Parameters:
    ----------
    input_tensor : torch.Tensor
        The input tensor.
    M, N, P, Q : int
        Hankel block dimensions.
    output_tensor : torch.Tensor
        The output tensor to be modified in-place.
    """
    output_tensor.fill_(0)

    for m in range(M):
        for n in range(N):
            for p in range(P):
                for q in range(Q):
                    output_tensor[..., m * P + p, n * Q + q] += input_tensor[
                        ..., m + n, p + q
                    ]

    return output_tensor


@jit.script
def block_hankel_adjoint_kernel(
    input_tensor: torch.Tensor,
    M: int,
    N: int,
    P: int,
    Q: int,
    output_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Adjoint operation for the block Hankel matrix (in-place modification of
    output_tensor).

    Parameters:
    ----------
    input_tensor : torch.Tensor
        The input tensor.
    M, N, P, Q : int
        Hankel block dimensions.
    output_tensor : torch.Tensor
        The output tensor to be modified in-place.
    """
    output_tensor.fill_(0)

    for m in range(M):
        for n in range(N):
            for p in range(P):
                for q in range(Q):
                    output_tensor[..., m + n, p + q] += input_tensor[
                        ..., m * P + p, n * Q + q
                    ]

    return output_tensor


@jit.script
def block_hankel_normal_kernel(
    input_tensor: torch.Tensor,
    M: int,
    N: int,
    P: int,
    Q: int,
    output_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Normal operation for the block Hankel matrix (in-place modification of
    output_tensor).

    Parameters:
    ----------
    input_tensor : torch.Tensor
        The input tensor.
    M, N, P, Q : int
        Hankel block dimensions.
    output_tensor : torch.Tensor
        The output tensor to be modified in-place.
    """
    output_tensor.fill_(0)

    for m in range(M):
        for n in range(N):
            for p in range(P):
                for q in range(Q):
                    output_tensor[..., m + n, p + q] += input_tensor[..., m + n, p + q]

    return output_tensor


def block_hankel_forward(
    input_tensor: torch.Tensor,
    shape: Tuple[int, ...],
    normalize: bool = False,
) -> torch.Tensor:
    """
    Perform the forward block Hankel transform.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        The input tensor.
    shape : Tuple[int, ...]
        The shape of the tensor.
    normalize : bool
        Whether to normalize the output.

    Returns:
    -------
    torch.Tensor
        The transformed tensor.
    """
    M, N, P, Q = block_hankel_shape(shape[-2:])
    norm_factor = torch.sqrt(
        torch.tensor(
            min(M * P, N * Q), dtype=input_tensor.dtype, device=input_tensor.device
        )
    )

    output_tensor = torch.empty(
        *input_tensor.shape[:-2],
        M * P,
        N * Q,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    output_tensor = block_hankel_forward_kernel(input_tensor, M, N, P, Q, output_tensor)

    if normalize:
        output_tensor /= norm_factor

    return output_tensor


def block_hankel_adjoint(
    input_tensor: torch.Tensor,
    shape: Tuple[int, ...],
    normalize: bool = False,
) -> torch.Tensor:
    """
    Perform the adjoint block Hankel transform.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        The input tensor.
    shape : Tuple[int, ...]
        The shape of the tensor.
    normalize : bool
        Whether to normalize the output.

    Returns:
    -------
    torch.Tensor
        The adjoint transformed tensor.
    """
    M, N, P, Q = block_hankel_shape(shape[-2:])
    norm_factor = torch.sqrt(
        torch.tensor(
            min(M * P, N * Q), dtype=input_tensor.dtype, device=input_tensor.device
        )
    )

    output_tensor = torch.zeros(
        *input_tensor.shape[:-2],
        M + N - 1,
        P + Q - 1,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    output_tensor = block_hankel_adjoint_kernel(input_tensor, M, N, P, Q, output_tensor)

    if normalize:
        output_tensor /= norm_factor

    return output_tensor


def block_hankel_normal(
    input_tensor: torch.Tensor,
    shape: Tuple[int, ...],
    normalize: bool = False,
) -> torch.Tensor:
    """
    Perform the normal block Hankel transform.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        The input tensor.
    shape : Tuple[int, ...]
        The shape of the tensor.
    normalize : bool
        Whether to normalize the output.

    Returns:
    -------
    torch.Tensor
        The normal transformed tensor.
    """
    M, N, P, Q = block_hankel_shape(shape[-2:])
    norm_factor = min(M * P, N * Q)

    output_tensor = torch.zeros_like(
        input_tensor, dtype=input_tensor.dtype, device=input_tensor.device
    )

    output_tensor = block_hankel_normal_kernel(input_tensor, M, N, P, Q, output_tensor)

    if normalize:
        output_tensor /= norm_factor

    return output_tensor
