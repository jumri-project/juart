from typing import Callable, Optional

import torch

from ..conopt.linops.channel import ChannelOperator
from ..conopt.linops.tf import TransferFunctionOperator
from ..conopt.proxalgs.cg import LinearSolver


class SENSE:
    def __init__(
        self,
        coil_sensitivities: torch.Tensor,
        regridded_data: torch.Tensor,
        transfer_function: torch.Tensor,
        channel_normalize: bool = True,
        maxiter: int = 15,
        verbose: bool = False,
        callback: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the SENSE solver.

        Parameters
        ----------
        coil_sensitivities : torch.Tensor
            Coil sensitivity maps used for sensitivity encoding (SENSE).
        regridded_data : torch.Tensor
            Regridded k-space data after Fourier and inverse Fourier
            transformations.
        transfer_function : torch.Tensor
            Transfer function used in the frequency domain, representing the
            encoding operator for the system.
        shape : tuple of ints
            Shape of the data in the form (nX, nY, nZ, nS, nTI, nTE), where nX,
            nY, nZ are spatial dimensions, nS is the number of channels, and
            nTI, nTE are the number of inversion and echo times, respectively.
        maxiter : int, optional
            Number of  iterations for solving the linear system using the
            conjugate gradient method (default is 15).
        callback : callable, optional
            Callback function for monitoring convergence during optimization
            (default is None).

        Notes
        -----

        References
        ----------
        """

        num_channels = coil_sensitivities.shape[0]
        shape = regridded_data.shape

        channel_operator = ChannelOperator(
            coil_sensitivities,
            regridded_data.shape,
            normalize=channel_normalize,
            device=device,
        )

        transfer_function_operator = TransferFunctionOperator(
            transfer_function,
            (num_channels,) + shape,
            axes=(1, 2),
            device=device,
        )

        self.solver = LinearSolver(
            regridded_data,
            channel_operator.H @ transfer_function_operator @ channel_operator,
            maxiter=maxiter,
            verbose=verbose,
        )

    def solve(self) -> torch.Tensor:
        """
        Solves the optimization problem using the CG solver.

        Returns:
        - The solution of the optimization problem.
        """
        return self.solver.solve()
