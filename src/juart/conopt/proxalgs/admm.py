from typing import Callable, Optional, Union

import torch
from tqdm import tqdm

from ..functional import norm
from ..linops import LinearOperator


class ADMM:
    """
    Alternating Direction Method of Multipliers (ADMM) using PyTorch.

    This implementation follows the algorithm described in [1].

    References:
    -----------
    .. [1] S. Boyd, “Distributed Optimization and Statistical Learning via the
           Alternating Direction Method of Multipliers,” Foundations and
           Trends® in Machine Learning, vol. 3, no. 1, pp. 1–122, 2010.
    """

    def __init__(
        self,
        prox_h: Callable,
        prox_g: Callable,
        A: Union[torch.Tensor, LinearOperator],
        B: Union[torch.Tensor, LinearOperator],
        c: Union[int, torch.Tensor],
        maxiter: int = 200,
        eps_abs: float = 0.0,
        eps_rel: float = 1e-9,
        tau: float = 1.0,
        verbose: bool = True,
        callback: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the ADMM solver.

        Parameters:
        ----------
        prox_h : Callable
            Proximal operator for the h function.
        prox_g : Callable
            Proximal operator for the g function.
        A : torch.Tensor or LinearOperator
            Linear operator A.
        B : torch.Tensor or LinearOperator
            Linear operator B.
        c : int or torch.Tensor
            The right-hand side vector.
        maxiter : int, optional
            Maximum number of iterations (default is 200).
        eps_abs : float, optional
            Absolute tolerance for convergence (default is 0.0).
        eps_rel : float, optional
            Relative tolerance for convergence (default is 1e-9).
        tau : float, optional
            Over-relaxation parameter (default is 1.0).
        verbose : bool, optional
            Whether to print progress information (default is True).
        callback : Callable, optional
            Function to call after each iteration (default is None).
        device : str, optional
            Device to run on, either 'cpu' or 'cuda' (default is 'cpu').
        """
        self.device = device
        self.dtype = torch.float32

        self.prox_h = prox_h
        self.prox_g = prox_g

        self.A = A.to(self.device) if isinstance(A, torch.Tensor) else A
        self.B = B.to(self.device) if isinstance(B, torch.Tensor) else B
        self.c = torch.as_tensor(c, device=self.device)

        self.maxiter = maxiter
        self.verbose = verbose
        self.external_callback = callback
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

        z = torch.zeros(self.B.shape[1], dtype=self.dtype, device=self.device)
        u = torch.zeros(self.A.shape[0], dtype=self.dtype, device=self.device)

        self.results = {
            "status": 2,
            "message": "Maximum number of iterations exceeded",
            "primal_residual": [],
            "dual_residual": [],
            "eps_prim": [],
            "eps_dual": [],
            "relative_residual": [],
            "tau": tau,
            "iteration": [],
            "v": z,
            "lamda": u,
        }

    def solve(self) -> dict:
        """
        Solve the ADMM problem.

        Returns:
        -------
        results : dict
            Dictionary containing the solution and convergence information.
        """
        z = self.results["v"]
        u = self.results["lamda"]
        tau = self.results["tau"]

        Bz = self.B @ z

        log = tqdm(
            total=self.maxiter,
            desc="ADMM",
            disable=(not self.verbose),
        )

        for iteration in range(self.maxiter):
            # ----------------------------------------------------------------
            # Update x, z, and u
            # ----------------------------------------------------------------
            Ax = self.A @ self.prox_h(self.c - Bz + u, tau)
            z = self.prox_g(self.c - Ax + u, tau)
            Bz_0 = Bz.clone()
            Bz = self.B @ z
            u = u + self.c - Ax - Bz

            # ----------------------------------------------------------------
            # Convergence analysis
            # ----------------------------------------------------------------
            r_norm = norm(self.c - Ax - Bz)
            d_norm = norm(tau * self.A.T @ (Bz - Bz_0))

            eps_prim = self.eps_abs * torch.sqrt(
                torch.tensor(
                    self.A.shape[0],
                    device=self.device,
                )
            ) + self.eps_rel * torch.max(
                torch.tensor(
                    [norm(Ax), norm(Bz), norm(self.c)],
                    device=self.device,
                )
            )

            eps_dual = self.eps_abs * torch.sqrt(
                torch.tensor(
                    self.A.shape[1],
                    device=self.device,
                )
            ) + self.eps_rel * norm(tau * self.A.T @ u)

            rel_norm = self.eps_rel * torch.max(
                torch.tensor(
                    [r_norm / eps_prim, d_norm / eps_dual],
                    device=self.device,
                )
            )

            # ----------------------------------------------------------------
            # Print progress
            # ----------------------------------------------------------------
            str_out = "[ADMM] "
            str_out += f"Iter: {iteration:0>{len(str(self.maxiter))}} "
            str_out += f"Prim Res: {r_norm.item():.2E} "
            str_out += f"Dual Res: {d_norm.item():.2E} "
            str_out += f"Rel Res: {rel_norm.item():.2E}"

            log.set_description_str(str_out)
            log.update(1)

            # ----------------------------------------------------------------
            # Log Progress
            # ----------------------------------------------------------------
            self.results["primal_residual"].append(r_norm.item())
            self.results["dual_residual"].append(d_norm.item())
            self.results["eps_prim"].append(eps_prim.item())
            self.results["eps_dual"].append(eps_dual.item())
            self.results["relative_residual"].append(rel_norm.item())

            self.results["iteration"].append(iteration)
            self.results["v"] = z
            self.results["lamda"] = u

            # ----------------------------------------------------------------
            # Run external callback and check external termination criteria
            # ----------------------------------------------------------------
            if self.external_callback is not None:
                if self.external_callback(self.results):
                    self.results["status"] = 0
                    self.results["message"] = (
                        f"[ADMM] Terminated after {iteration} iterations."
                    )
                    break

            # ----------------------------------------------------------------
            # Check convergence criteria
            # ----------------------------------------------------------------
            if rel_norm < self.eps_rel:
                self.results["status"] = 0
                self.results["message"] = (
                    f"[ADMM] Converged after {iteration} iterations."
                )
                break

        # log.write(self.results["message"])

        return self.results
