from typing import Tuple

import torch
from torch import jit


@jit.script
def block_hankel_shape(
    shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    M = shape[0] // 2 + 1
    N = shape[0] - M + 1

    P = shape[1] // 2 + 1
    Q = shape[1] - P + 1

    return M, N, P, Q


@jit.script
def block_hankel_forward_kernel(
    x: torch.Tensor,
    M: int,
    N: int,
    P: int,
    Q: int,
    y: torch.Tensor,
) -> None:
    y.fill_(0)

    for m in range(M):
        for n in range(N):
            for p in range(P):
                for q in range(Q):
                    y[..., m * P + p, n * Q + q] += x[..., m + n, p + q]

    return


@jit.script
def block_hankel_foward(
    x: torch.Tensor,
    shape: Tuple[int, int],
) -> torch.Tensor:
    M, N, P, Q = block_hankel_shape(shape)

    y = torch.empty(
        x.shape[:-2] + (M * P, N * Q),
        device=x.device,
        dtype=x.dtype,
    )

    block_hankel_forward_kernel(x, M, N, P, Q, y)

    return y
