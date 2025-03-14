from typing import Callable, Optional, Tuple

import torch

from ..conopt.linops.composite import ConcatOperator, SumOperator
from ..conopt.linops.identity import IdentityOperator
from ..conopt.linops.system import SystemMatrixNormalOperator, SystemMatrixOperator
from ..conopt.linops.tf import OversampledTransferFunctionNormalOperator
from ..conopt.linops.wavelet import WaveletTransformOperator
from ..conopt.proxalgs.admm import ADMM
from ..conopt.proxops.composite import SeparableProximalOperator
from ..conopt.proxops.linear import LinearProximalSolver
from ..conopt.proxops.nuclear import SingularValueSoftThresholdingOperator
from ..conopt.proxops.taxicab import JointSoftThresholdingOperator


class SAKE(object):
    def __init__(
        self,
        regridded_data: torch.Tensor,
        transfer_function: torch.Tensor,
        lambda_wavelet: Optional[float] = None,
        lamda_system: Optional[float] = 1e-1,
        weight_wavelet: float = 1.0,
        weight_system: float = 1.0,
        tau: float = 1.0,
        wavelet_type: str = "db4",
        wavelet_level: int = 4,
        system_window: Tuple[int, int, int] = (6, 6, 6),
        inner_iter: int = 5,
        outer_iter: int = 500,
        callback: Optional[Callable] = None,
    ):
        nC, nX, nY, nZ, nS, nTI, nTE = regridded_data.shape

        lin_ops = []
        lin_ops_normal = []
        prox_ops = []

        # Define the OversampledTransferFunctionNormalOperator
        transfer_function_operator = OversampledTransferFunctionNormalOperator(
            transfer_function, (nC, nX, nY, nZ, nS, nTI, nTE), nonuniform_axes=(1, 2)
        )

        if lambda_wavelet is not None:
            # Define the wavelet transform operator
            wavelet_operator = WaveletTransformOperator(
                (nC, nX, nY, nZ, nS, nTI, nTE),
                axes=(1, 2),
                wavelet=wavelet_type,
                level=wavelet_level,
            )
            lin_ops.append(-weight_wavelet * wavelet_operator)
            lin_ops_normal.append(
                weight_wavelet**2 * IdentityOperator((nC, nX, nY, nZ, nS, nTI, nTE))
            )

            # Add the JointSoftThresholdingOperator to the proximal operators
            prox_ops.append(
                JointSoftThresholdingOperator(
                    wavelet_operator.adjoint_shape,
                    weight_wavelet * lambda_wavelet,
                    axes=(0, -1, -2),
                )
            )

        if lamda_system is not None:
            # Define the system matrix operator
            system_operator = SystemMatrixOperator(
                (nC, nX, nY, nZ, nS, nTI, nTE), system_window
            )
            lin_ops.append(-weight_system * system_operator)
            lin_ops_normal.append(
                weight_system**2
                * SystemMatrixNormalOperator(
                    (nC, nX, nY, nZ, nS, nTI, nTE), system_window
                )
            )

            # Add the SingularValueSoftThresholdingOperator to the proximal operators
            prox_ops.append(
                SingularValueSoftThresholdingOperator(
                    system_operator.adjoint_shape,
                    weight_system * lamda_system,
                    (5, 4, 3, 2, 1, 0),
                    (
                        nTE,
                        nTI,
                        nS,
                        system_operator.adjoint_shape[2],
                        system_operator.adjoint_shape[1]
                        * system_operator.adjoint_shape[0],
                    ),
                )
            )

        # Define the composite operators B and A
        self.concatenated_operator = ConcatOperator(lin_ops)
        self.identity_operator = IdentityOperator(
            self.concatenated_operator.shape[0] // 2
        )
        self.constant = torch.tensor(0.0, dtype=torch.float32)

        # Define the normal operator as the sum of linear operators
        self.concatenated_normal_operator = SumOperator(lin_ops_normal)

        # Define the proximal operators
        prox_f = SeparableProximalOperator(
            prox_ops,
            self.concatenated_operator.indices,
        )
        prox_g = LinearProximalSolver(
            regridded_data.view(torch.float32).ravel(),
            transfer_function_operator,
            self.concatenated_operator,
            self.concatenated_normal_operator,
            maxiter=inner_iter,
        )

        # Initialize the ADMM solver
        self.solver = ADMM(
            prox_f.solve,
            prox_g.solve,
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
        Solve the optimization problem using ADMM.
        """
        return self.solver.solve()["v"]


def sake(
    regridded_data: torch.Tensor,
    transfer_function: torch.Tensor,
    lamda_system: float = 1e-1,
    inner_iter: int = 1,
    outer_iter: int = 100,
) -> torch.Tensor:
    """
    Wrapper function for the SAKE solver.

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
    solver = SAKE(
        regridded_data.clone(),
        transfer_function.clone(),
        lamda_system=lamda_system,
        inner_iter=inner_iter,
        outer_iter=outer_iter,
    )

    # Solve the optimization problem
    coil_images = solver.solve()

    # Reshape the solution to the original data shape
    coil_images = coil_images.view(torch.complex64).reshape(regridded_data.shape)

    return coil_images
