from typing import Union

import torch

from ..linops import LinearOperator


def conjugate_gradient(
    A: Union[torch.Tensor, LinearOperator],
    b: torch.Tensor,
    x: torch.Tensor,
    maxiter: int,
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

    for _ in range(maxiter):
        Ap = A @ p
        alpha = rsold / torch.dot(p.flatten(), Ap.flatten())
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r.flatten(), r.flatten())
        if torch.sqrt(rsnew) < 1e-10:  # Early exit if residual is sufficiently small
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


class LinearProximalSolver:
    """
    Linear Proximal Solver using the Conjugate Gradient method with PyTorch.

    This solver handles linear systems of the form (AHA + BHB) @ x = AHd + B^T @ v.
    """

    def __init__(
        self,
        AHd: torch.Tensor,
        AHA: Union[torch.Tensor, LinearOperator],
        B: Union[torch.Tensor, LinearOperator],
        BHB: Union[torch.Tensor, LinearOperator],
        maxiter: int = 1500,
        verbose: bool = False,
    ):
        """
        Initialize the Linear Proximal Solver.

        Parameters:
        ----------
        AHd : torch.Tensor
            Precomputed product A @ H @ d.
        AHA : torch.Tensor or LinearOperator
            Precomputed product A @ H @ A.
        B : torch.Tensor or LinearOperator
            The matrix B or an operator that computes B @ x.
        BHB : torch.Tensor or LinearOperator
            Precomputed product B @ H @ B.
        maxiter : int, optional
            Maximum number of iterations for the conjugate gradient solver
            (default is 1500).
        verbose : bool, optional
            If True, prints progress information (default is False).
        """
        self.AHd = AHd
        self.AHA = AHA
        self.B = B
        self.BHB = BHB

        # Initial guess for the solution x
        self.x = torch.zeros_like(AHd)
        self.maxiter = maxiter
        self.verbose = verbose

    def solve(
        self,
        v: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        """
        Solves the linear proximal problem using the conjugate gradient method.

        Parameters:
        ----------
        v : torch.Tensor
            The input tensor.
        rho : float
            Regularization parameter.

        Returns:
        -------
        x : torch.Tensor
            The solution tensor to the linear system.
        """
        # Solve the system (AHA + BHB) @ x = AHd + B^T @ v
        rhs = self.AHd + self.B.T @ v
        A_combined = self.AHA + self.BHB
        self.x = conjugate_gradient(A_combined, rhs, self.x, self.maxiter)

        return self.x
