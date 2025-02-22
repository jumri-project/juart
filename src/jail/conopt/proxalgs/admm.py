from typing import Callable, Optional, Tuple, Union

import torch

from ..aux import norm
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
        verbose: bool = True,
        callback: Optional[Callable] = None,
        eps_abs: float = 0.0,
        eps_rel: float = 1e-9,
        tau: float = 1.0,
        verbose_fmt: Tuple[str] = (
            "Iteration",
            "Primal Residual",
            "Dual Residual",
            "Relative Residual",
            "Tau",
        ),
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
        verbose : bool, optional
            Whether to print progress information (default is True).
        callback : Callable, optional
            Function to call after each iteration (default is None).
        eps_abs : float, optional
            Absolute tolerance for convergence (default is 0.0).
        eps_rel : float, optional
            Relative tolerance for convergence (default is 1e-9).
        tau : float, optional
            Over-relaxation parameter (default is 1.0).
        verbose_fmt : list of str, optional
            Formatting options for the verbosity output
            (default is standard ADMM metrics).
        device : str, optional
            Device to run on, either 'cpu' or 'cuda' (default is 'cpu').
        """
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.prox_h = prox_h
        self.prox_g = prox_g

        self.A = A.to(self.device) if isinstance(A, torch.Tensor) else A
        self.B = B.to(self.device) if isinstance(B, torch.Tensor) else B
        self.c = torch.as_tensor(c, device=self.device)

        self.maxiter = maxiter
        self.verbose = verbose
        self.verbose_fmt = verbose_fmt
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

        if self.verbose:
            print("[ADMM] Initialization done.")

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
                torch.tensor(self.A.shape[0], device=self.device)
            ) + self.eps_rel * torch.max(
                torch.tensor(
                    [norm(Ax), norm(Bz), norm(self.c)],
                    device=self.device,
                )
            )

            eps_dual = self.eps_abs * torch.sqrt(
                torch.tensor(self.A.shape[1], device=self.device)
            ) + self.eps_rel * norm(tau * self.A.T @ u)

            rel_norm = self.eps_rel * torch.max(
                torch.tensor([r_norm / eps_prim, d_norm / eps_dual], device=self.device)
            )

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
            # Print progress and execute callback
            # ----------------------------------------------------------------
            terminate = self.callback()
            if terminate:
                self.results["status"] = 0
                self.results["message"] = (
                    f"[ADMM] Terminated after {iteration} iterations."
                )
                break

            # ----------------------------------------------------------------
            # Check termination criteria
            # ----------------------------------------------------------------
            if rel_norm < self.eps_rel:
                self.results["status"] = 0
                self.results["message"] = (
                    f"[ADMM] Converged after {iteration} iterations."
                )
                break

        if self.verbose:
            print(self.results["message"])

        return self.results

    def callback(self) -> bool:
        """
        Internal callback of the ADMM loop, executed after each iteration.

        Returns:
        -------
        terminate : bool
            Whether the ADMM loop should terminate early.
        """
        terminate = False
        if self.external_callback is not None:
            terminate = self.external_callback(self.results)

        if self.verbose:
            iteration = self.results["iteration"][-1]
            str_iteration = f"{iteration:0>{len(str(self.maxiter))}}"
            str_norm_prim = f"{self.results['primal_residual'][-1]:.2E}"
            str_norm_dual = f"{self.results['dual_residual'][-1]:.2E}"
            str_norm_rel = f"{self.results['relative_residual'][-1]:.2E}"

            str_out = "[ADMM] "

            if "Iteration" in self.verbose_fmt:
                str_out += f"Iter: {str_iteration} "
            if "Primal Residual" in self.verbose_fmt:
                str_out += f"Prim Res: {str_norm_prim} "
            if "Dual Residual" in self.verbose_fmt:
                str_out += f"Dual Res: {str_norm_dual} "
            if "Relative Residual" in self.verbose_fmt:
                str_out += f"Rel Res: {str_norm_rel} "

            print(str_out)

        return terminate
