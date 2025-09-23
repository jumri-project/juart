from typing import List, Union

import torch
from tqdm import tqdm

from ..linops import LinearOperator


def conjugate_gradient(
    A: Union[torch.Tensor, LinearOperator],
    b: torch.Tensor,
    x: torch.Tensor,
    residual: List[int],
    maxiter: int,
    eps: float = 1e-10,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Conjugate Gradient solver using PyTorch.

    Solves the system A @ x = b using the conjugate gradient method, which is
    particularly useful for large sparse linear systems.

    Parameters:
    ----------
    A : torch.Tensor or LinearOperator
        A tensor representing the matrix A or a LinearOperator that computes the
        matrix-vector product A @ x.
    b : torch.Tensor
        The right-hand side vector.
    x : torch.Tensor
        The initial guess for the solution.
    maxiter : int
        Maximum number of iterations for the solver.

    Returns:
    -------
    x : torch.Tensor
        The solution tensor to the system A @ x = b.
    """
    r = b - A @ x
    p = r.clone()
    rsold = torch.dot(r.flatten(), r.flatten())

    log = tqdm(
        total=maxiter,
        desc="CG",
        disable=(not verbose),
    )

    for iteration in range(maxiter):
        Ap = A @ p
        alpha = rsold / torch.dot(p.flatten(), Ap.flatten())
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r.flatten(), r.flatten())
        p = r + (rsnew / rsold) * p
        rsold = rsnew

        str_out = "[CG] "
        str_out += f"Iter: {(iteration + 1):0>{len(str(maxiter))}} "
        str_out += f"Res: {rsnew:.2E} "
        log.set_description_str(str_out)
        log.update(1)

        residual.append(rsnew)

        if torch.sqrt(rsnew) < eps:
            break

    return x, residual


class LinearSolver:
    def __init__(
        self,
        AHd: torch.Tensor,
        AHA: Union[torch.Tensor, LinearOperator],
        maxiter: int,
        eps: float = 1e-10,
        verbose: bool = False,
    ):
        self.AHd = AHd.view(torch.float32).ravel()
        self.AHA = AHA

        self.maxiter = maxiter
        self.x = torch.zeros_like(self.AHd)
        self.residual = []
        self.verbose = verbose

    def solve(
        self,
    ) -> torch.Tensor:
        self.x, self.residual = conjugate_gradient(
            self.AHA,
            self.AHd,
            self.x,
            self.residual,
            self.maxiter,
            verbose=self.verbose,
        )

        return self.x
