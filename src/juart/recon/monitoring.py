import time
from typing import Optional

import torch

from ..conopt.functional import norm


class ConvergenceMonitor:
    """
    A class to monitor the convergence of the ADMM optimization algorithm.
    Tracks the norm of the solution, the time per iteration, and checks for termination
    based on a target norm.
    """

    def __init__(
        self,
        target_tensor: torch.Tensor,
        support_mask: torch.Tensor,
        target_norm: float = -1,
        logfile: Optional[str] = None,
        dtype: torch.dtype = torch.complex64,
    ):
        """
        Initializes the convergence monitor.

        Parameters:
        ----------
        target_tensor : torch.Tensor
            The reference tensor to compare against.
        support_mask : torch.Tensor
            The support mask tensor that defines the region of interest for the norm
            calculation.
        target_norm : float, optional
            The norm threshold for termination (default is -1, meaning no termination).
        logfile : str, optional
            Path to a logfile (not currently implemented, default is None).
        dtype : torch.dtype, optional
            The data type for computations (default is torch.complex64).
        """
        self.target_tensor = target_tensor
        self.target_norm_value = norm(target_tensor)
        self.norm = []
        self.time = []
        self.support_mask = support_mask
        self.target_norm = target_norm
        self.dtype = dtype

        print("[Convergence Monitor] Initialization complete.")

    def callback(self, results: dict) -> bool:
        """
        Callback function to monitor convergence during ADMM optimization.

        Parameters:
        ----------
        results : dict
            A dictionary containing the current solution tensor 'v' from ADMM.

        Returns:
        -------
        terminate : bool
            Indicates whether to terminate the solver based on the convergence criteria.
        """
        terminate = False

        # Extract the current solution tensor 'v' from ADMM results
        current_solution = results["v"]

        # Reshape and cast the current solution to match the target tensor
        current_solution = current_solution.view(self.dtype).reshape(
            self.target_tensor.shape
        )

        # Calculate the relative norm of the difference within the support region
        relative_norm = (
            norm((current_solution - self.target_tensor) * self.support_mask)
            / self.target_norm_value
        )
        self.norm.append(relative_norm.item())

        # Track the elapsed time for each iteration
        self.time.append(time.time())
        time_per_iteration = 0 if len(self.time) < 2 else self.time[-1] - self.time[-2]

        # Check if the norm is below the target norm for termination
        if self.norm[-1] < self.target_norm:
            terminate = True

        # Log the current status
        print(
            f"[Convergence Monitor] Norm: {self.norm[-1]:10.3e} \t "
            f"Time per Iteration: {time_per_iteration:10.3e}"
        )

        return terminate
