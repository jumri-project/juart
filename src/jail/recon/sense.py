from typing import Callable, Optional

import torch

from ..conopt.linops.channel import ChannelOperator
from ..conopt.linops.composite import ConcatOperator, SumOperator
from ..conopt.linops.identity import IdentityOperator
from ..conopt.linops.tf import TransferFunctionNormalOperator
from ..conopt.linops.wavelet import WaveletTransformOperator
from ..conopt.proxalgs.admm import ADMM
from ..conopt.proxops.composite import SeparableProximalOperator
from ..conopt.proxops.linear import LinearProximalSolver
from ..conopt.proxops.taxicab import JointSoftThresholdingOperator


class SENSE(object):
    def __init__(
        self,
        coil_sensitivities: torch.Tensor,
        regridded_data: torch.Tensor,
        transfer_function: torch.Tensor,
        lambda_wavelet: Optional[float] = 1e-4,
        weight_wavelet: float = 1.0,
        tau: float = 1.0,
        channel_normalize: bool = False,
        wavelet_type: str = "db4",
        wavelet_level: int = 4,
        inner_iter: int = 10,
        outer_iter: int = 20,
        callback: Optional[Callable] = None,
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
        lambda_wavelet : float, optional
            Regularization parameter for the wavelet term to encourage sparsity
            in the wavelet domain (default is 1e-3).
        weight_wavelet : float, optional
            Weight assigned to the wavelet regularizer (default is 1.0).
        tau : int, optional
            Step size parameter for the ADMM solver (default is 1).
        channel_normalize : bool, optional
            Whether to normalize the channel operator (default is True).
        wavelet_type : str, optional
            Type of wavelet used in the wavelet transform (default is 'db4').
        wavelet_level : int, optional
            Number of wavelet decomposition levels (default is 4).
        inner_iter : int, optional
            Number of inner iterations for solving the linear system using the
            conjugate gradient method (default is 5).
        outer_iter : int, optional
            Maximum number of outer iterations for the ADMM solver
            (default is 500).
        callback : callable, optional
            Callback function for monitoring convergence during optimization
            (default is None).

        Notes
        -----

        References
        ----------
        """

        nX, nY, nZ, nS, nTI, nTE = regridded_data.shape
        nC = coil_sensitivities.shape[0]  # Number of channels

        shape = regridded_data.shape

        lin_ops = []
        lin_ops_normal = []
        prox_ops = []

        # Define the channel operator
        channel_operator = ChannelOperator(
            coil_sensitivities, shape, normalize=channel_normalize
        )

        # Define the oversampled transfer function normal operator
        transfer_function_operator = TransferFunctionNormalOperator(
            transfer_function, (nC, nX, nY, nZ, nS, nTI, nTE), axes=(1, 2)
        )

        if lambda_wavelet is not None:
            # Define the wavelet transform operator
            wavelet_operator = WaveletTransformOperator(
                shape, axes=(0, 1), wavelet=wavelet_type, level=wavelet_level
            )
            lin_ops.append(-weight_wavelet * wavelet_operator)
            lin_ops_normal.append(weight_wavelet**2 * IdentityOperator(shape))
            prox_ops.append(
                JointSoftThresholdingOperator(
                    wavelet_operator.adjoint_shape,
                    weight_wavelet * lambda_wavelet,
                    axes=(-1, -2),
                )
            )

        # Define the composite operators
        self.concatenated_operator = ConcatOperator(lin_ops)
        self.identity_operator = IdentityOperator(
            self.concatenated_operator.shape[0] // 2
        )
        self.constant = torch.tensor(0.0, dtype=torch.float32)

        self.concatenated_normal_operator = SumOperator(lin_ops_normal)

        # Define the proximal operators
        self.prox_f = SeparableProximalOperator(
            prox_ops,
            self.concatenated_operator.indices,
        )
        self.prox_g = LinearProximalSolver(
            regridded_data.view(torch.float32).ravel(),
            channel_operator.H @ transfer_function_operator @ channel_operator,
            self.concatenated_operator,
            self.concatenated_normal_operator,
            maxiter=inner_iter,
        )

        # Initialize the ADMM solver
        self.solver = ADMM(
            self.prox_f.solve,
            self.prox_g.solve,
            self.identity_operator,
            self.concatenated_operator,
            self.constant,
            maxiter=outer_iter,
            verbose=False,
            callback=callback,
            tau=tau,
        )

    def solve(self) -> torch.Tensor:
        """
        Solves the optimization problem using the ADMM solver.

        Returns:
        - v: The solution of the optimization problem.
        """
        return self.solver.solve()["v"]


def sense(
    coil_sensitivities: torch.Tensor,
    regridded_data: torch.Tensor,
    transfer_function: torch.Tensor,
    lambda_wavelet: float = 1e-4,
    channel_normalize: bool = False,
    inner_iter: int = 10,
    outer_iter: int = 20,
) -> torch.Tensor:
    """
    Wrapper function for the SENSE solver.

    Parameters
    ----------
    regridded_data : torch.Tensor
        Input data tensor.
    H : torch.Tensor
        Transfer function.
    lamda_system : float
        Regularization parameter for the system matrix.
    inner_iter : int
        Number of inner iterations for the ADMM solver.
    outer_iter : int
        Number of outer iterations for the ADMM solver.

    Returns
    -------
    torch.Tensor
        Reconstructed data.
    """
    # Initialize the SAKE solver
    solver = SENSE(
        coil_sensitivities.clone(),
        regridded_data.clone(),
        transfer_function.clone(),
        lambda_wavelet=lambda_wavelet,
        channel_normalize=channel_normalize,
        inner_iter=inner_iter,
        outer_iter=outer_iter,
    )

    # Solve the optimization problem
    images = solver.solve()

    # Reshape the solution to the original data shape
    images = images.view(torch.complex64).reshape(regridded_data.shape)

    return images
