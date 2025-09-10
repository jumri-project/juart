from typing import Callable, Optional, Union

import torch

from ..conopt.functional.fourier import (
    fourier_transform_adjoint,
    nonuniform_fourier_transform_adjoint,
)
from ..conopt.linops.channel import ChannelOperator
from ..conopt.linops.identity import IdentityOperator
from ..conopt.linops.tf import TransferFunctionOperator
from ..conopt.proxalgs.cg import LinearSolver
from ..conopt.tfs.fourier import (
    nonuniform_transfer_function,
)


def cgsense(
    ksp: torch.Tensor,
    ktraj: torch.Tensor,
    coilsens: torch.Tensor,
    maxiter: int = 20,
    l2_reg: float = 0.0,
    channel_normalize: bool = True,
    verbose: bool = False,
    device: Optional[torch.device] = None,
    callback: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Solve the SENSE reconstruction problem using the conjugate gradient method.

    Parameters
    ----------
    ksp : torch.Tensor, shape (C, N, S, ...)
        Raw k-space data to be reconstructed, where C is the number of channels,
        N is the number of samples, S is the number of slices/slabs
        and ... represents additional dimensions.
    ktraj : torch.Tensor, shape (D, N, S, ...)
        Trajectory of the k-space sampling, where D is the number of dimensions
        N is the number of samples, S is the number of slices/slabs
        and ... represents additional dimensions.
    coilsens : torch.Tensor, shape (C, X, Y, Z, S)
        Coil sensitivity maps used for sensitivity encoding (SENSE), where C is the
        number of channels, X, Y and Z are the spatial dimensions
        and S is the number of slices/slabs.
    maxiter : int, optional
        Maximum number of iterations for the conjugate gradient solver (default is 20).
    l2_reg: flaot, optional
        L2 regularization parameter (default is 0, no regularization).
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
    """
    num_dim, num_col, num_slc, *add_axes = ktraj.shape
    num_cha, num_x, num_y, num_z, num_slc = coilsens.shape

    if num_dim == 2:
        modes = (num_x, num_y)
    elif num_dim == 3:
        modes = (num_x, num_y, num_z)
    else:
        raise ValueError("ktraj must be 2D or 3D.")

    # Collaps additional dimensions
    ksp = ksp.reshape(num_cha, num_col, num_slc, -1)
    ktraj = ktraj.reshape(num_dim, num_col, num_slc, -1)

    regridded_data = nonuniform_fourier_transform_adjoint(
        k=ktraj,
        x=ksp,
        n_modes=modes,
    )

    transfer_function = nonuniform_transfer_function(
        k=ktraj,
        data_shape=(1, *regridded_data.shape[1:]),
    )

    regridded_data = torch.sum(
        torch.conj(coilsens)[..., None] * regridded_data, dim=0, keepdim=True
    )

    # Create SENSE solver instance
    sense_solver = SENSE(
        coil_sensitivities=coilsens,
        regridded_data=regridded_data,
        transfer_function=transfer_function,
        channel_normalize=channel_normalize,
        maxiter=maxiter,
        axes=((1, 2) if num_dim == 2 else (1, 2, 3)),
        l2_reg=l2_reg,
        verbose=verbose,
        callback=callback,
        device=device,
    )

    # Solve the SENSE reconstruction problem
    img = sense_solver.solve()
    img = img.view(torch.complex64).reshape(1, num_x, num_y, num_z, num_slc, *add_axes)

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
        axes: Union[tuple[int, int], tuple[int, int, int]] = (1, 2),
        maxiter: int = 15,
        lambda_ridge: float = 0.0,
        verbose: bool = False,
        channel_normalize: bool = True,
        callback: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the SENSE solver.

        Parameters
        ----------
        coil_sensitivities : torch.Tensor, shape (C, X, Y, Z, S)
            Coil sensitivity maps used for sensitivity encoding (SENSE).
        regridded_data : torch.Tensor, shape (1, X, Y, Z, S, M)
            Regridded k-space data after adjoint (non-uniform) fourier transform
            and coil combination.
            Dimensions are (1, X, Y, Z, S, M),
            where 1 is the number of (combined) channels,
            X, Y, Z are the spatial dimensions,
            S is the number of slices/slabs,
            and M represents additional batch dimensions.
        transfer_function : torch.Tensor, shape (1, X, Y, Z, S, M)
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
        shape = regridded_data.shape[1:]

        channel_operator = ChannelOperator(
            coil_sensitivities,
            (num_channels,) + shape,
            normalize=channel_normalize,
            device=device,
        )

        transfer_function_operator = TransferFunctionOperator(
            transfer_function,
            (num_channels,) + shape,
            axes=axes,
            device=device,
        )

        AhA = channel_operator.H @ transfer_function_operator @ channel_operator

        identity_operator = IdentityOperator(
            input_shape=shape,
            device=device,
        )

        self.solver = LinearSolver(
            regridded_data,
            AhA + lambda_ridge * identity_operator,
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
