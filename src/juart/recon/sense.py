from typing import Callable, Optional

import torch

from ..conopt.functional.fourier import (
    fourier_transform_adjoint,
    nonuniform_fourier_transform_adjoint,
)
from ..conopt.linops.channel import ChannelOperator
from ..conopt.linops.tf import TransferFunctionOperator
from ..conopt.proxalgs.cg import LinearSolver
from ..conopt.tfs.fourier import (
    nonuniform_transfer_function,
)


def cgsense(
    ksp_data: torch.Tensor,
    ktraj: torch.Tensor,
    coil_sensitivities: torch.Tensor,
    maxiter: int = 20,
    channel_normalize: bool = True,
    verbose: bool = False,
    device: Optional[torch.device] = None,
    callback: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Solve the SENSE reconstruction problem using the conjugate gradient method.

    Parameters
    ----------
    ksp_data : torch.Tensor, shape (nC, nN)
        Raw k-space data to be reconstructed, where nC is the number of channels,
        nN is the number of spatial points, and ... represents additional dimensions.
    ktraj : torch.Tensor, shape (nD, nN)
        Trajectory of the k-space sampling, where nD is the number of dimensions
        nN is the number of spatial point, and ... represents additional dimensions.
        Trajectory of the k-space sampling.
    coil_sensitivities : torch.Tensor, shape (nC, nX, nY, nZ)
        Coil sensitivity maps used for sensitivity encoding (SENSE), where nC is the
        number of channels and nX, nY, nZ are the spatial dimensions.
    maxiter : int, optional
        Maximum number of iterations for the conjugate gradient solver (default is 20).
    channel_normalize : bool, optional
        Whether to normalize the channel sensitivities (default is True).
    verbose : bool, optional
        If True, print convergence information (default is False).
    device : torch.device, optional
        Device on which to perform the computation
        (default is None, which uses the current device).
    callback : callable, optional
        Callback function for monitoring convergence
        during optimization (default is None).


    NOTE: This function is under development and may not be fully functional yet.
    """
    num_dim, _ = ktraj.shape[:2]
    num_cha, num_x, num_y, num_z = coil_sensitivities.shape[:4]

    modes = [dim for dim in [num_x, num_y, num_z] if dim > 1]

    # Check channel dimensions
    if num_cha != ksp_data.shape[0]:
        raise ValueError(
            f"Number of channels in ksp_data ({ksp_data.shape[0]}) does not match "
            f"number of channels in coil_sensitivities ({num_cha})."
        )

    # Check spatial dimensions
    if (3 - len(modes)) > num_dim:
        raise ValueError(
            f"Number of dimensions in ktraj ({num_dim}) is less than the number of "
            f"spatial dimensions in coil_sensitivities ({3 - len(modes)})."
        )

    regridded_data = nonuniform_fourier_transform_adjoint(
        k=ktraj,
        x=ksp_data,
        n_modes=tuple(modes),
    )

    transfer_function = nonuniform_transfer_function(
        k=ktraj,
        data_shape=(1, *regridded_data.shape[1:]),
    )

    regridded_data = torch.sum(torch.conj(coil_sensitivities) * regridded_data, dim=0)

    # Create SENSE solver instance
    sense_solver = SENSE(
        coil_sensitivities=coil_sensitivities,
        regridded_data=regridded_data,
        transfer_function=transfer_function,
        channel_normalize=channel_normalize,
        maxiter=maxiter,
        verbose=verbose,
        callback=callback,
        device=device,
    )

    # Solve the SENSE reconstruction problem
    img = sense_solver.solve()
    img = img.view(torch.complex64).reshape(1, num_x, num_y, num_z)

    return img


def cart_cgsense(
    ksp: torch.Tensor,
    coil_sensitivities: torch.Tensor,
    channel_normalize: bool = True,
    maxiter: int = 20,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Solve the SENSE reconstruction problem using the conjugate gradient method
    for Cartesian k-space data.

    Parameters
    ----------
    ksp : torch.Tensor, shape (nC, nX, nY)
        Raw k-space data to be reconstructed, where nC is the number of channels,
        and nX, nY are the spatial dimensions.
    coil_sensitivities : torch.Tensor, shape (nC, nX, nY)
        Coil sensitivity maps used for sensitivity encoding (SENSE).
    channel_normalize : bool, optional
        Whether to normalize the channel sensitivities (default is True).
    maxiter : int, optional
        Maximum number of iterations for the conjugate gradient solver (default is 20).
    verbose : bool, optional
        If True, print convergence information (default is False).

    Returns
    -------
    img : torch.Tensor
        Reconstructed image
        with shape (1, nX, nY).
    """
    num_cha, num_x, num_y, num_z = coil_sensitivities.shape

    if ksp.shape[0] != num_cha:
        raise ValueError(
            f"Number of channels in ksp ({ksp.shape[0]}) does not match "
            f"number of channels in coil_sensitivities ({num_cha})."
        )

    transfer_function = torch.where(ksp[0, ...] != 0, 1, 0).to(ksp.device)
    transfer_function = transfer_function[None, ...]

    img_data = fourier_transform_adjoint(
        ksp,
        axes=(1, 2, 3),
    )

    regridded_data = torch.sum(torch.conj(coil_sensitivities) * img_data, dim=0)

    sense_operator = SENSE(
        coil_sensitivities=coil_sensitivities,
        regridded_data=regridded_data,
        transfer_function=transfer_function,
        channel_normalize=channel_normalize,
        maxiter=maxiter,
        verbose=verbose,
    )

    img = sense_operator.solve()
    img = img.view(torch.complex64).reshape(1, num_x, num_y)

    return img


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

        channel_operator = ChannelOperator(
            coil_sensitivities,
            (num_channels,) + regridded_data.shape,
            normalize=channel_normalize,
            device=device,
        )

        transfer_function_operator = TransferFunctionOperator(
            transfer_function,
            (num_channels,) + regridded_data.shape,
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
