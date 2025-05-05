from typing import Union

import torch

from ..linops import LinearOperator
from ..proxalgs.cg import conjugate_gradient


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
        """
        self.AHd = AHd.view(torch.float32).ravel()
        self.AHA = AHA
        self.B = B
        self.BHB = BHB

        self.x = torch.zeros_like(AHd)
        self.residual = []
        self.maxiter = maxiter

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

        self.x, self.residual = conjugate_gradient(
            self.AHA + rho * self.BHB,
            self.AHd + rho * self.B.T @ v.view(torch.float32).ravel(),
            self.x,
            self.residual,
            self.maxiter,
        )

        return self.x
