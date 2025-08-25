from typing import Callable, Optional, Tuple

import torch

from ..conopt.linops.channel import ChannelOperator
from ..conopt.linops.composite import ConcatOperator, SumOperator
from ..conopt.linops.hankel import BlockHankelNormalOperator, BlockHankelOperator
from ..conopt.linops.identity import IdentityOperator
from ..conopt.linops.shift import ShiftOperator
from ..conopt.linops.tf import TransferFunctionOperator
from ..conopt.linops.wavelet import WaveletTransformOperator
from ..conopt.proxalgs.admm import ADMM
from ..conopt.proxops.composite import SeparableProximalOperator
from ..conopt.proxops.linear import LinearProximalSolver
from ..conopt.proxops.nuclear import SingularValueSoftThresholdingOperator
from ..conopt.proxops.taxicab import JointSoftThresholdingOperator


class MIRAGE(object):
    def __init__(
        self,
        coil_sensitivities: torch.Tensor,
        regridded_data: torch.Tensor,
        transfer_function: torch.Tensor,
        lambda_wavelet: Optional[float] = 1e-3,
        lambda_hankel: Optional[float] = None,
        lambda_casorati: Optional[float] = None,
        weight_wavelet: float = 1.0,
        weight_hankel: float = 1.0,
        weight_casorati: float = 1.0,
        tau: float = 1.0,
        channel_normalize: bool = True,
        wavelet_type: str = "db4",
        wavelet_level: int = 4,
        casorati_window: Tuple[int, int] = (3, 3),
        cg_maxiter: int = 5,
        admm_maxiter: int = 500,
        verbose: bool = True,
        callback: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the MIRAGE solver for QRAGE, a model-based iterative
        reconstruction method tailored to ultrahigh field (UHF) quantitative MRI.

        MIRAGE reconstructs data acquired by the QRAGE sequence, which
        simultaneously maps multiple quantitative MRI parameters—water content, T1,
        T2*, and magnetic susceptibility—at UHF using structured low-rank matrix
        methods. The reconstruction minimizes local Block-Hankel and Casorati
        matrices to exploit prior knowledge of temporal signal evolution.

        Parameters
        ----------
        coil_sensitivities : torch.Tensor
            Coil sensitivity maps used for sensitivity encoding (SENSE).
        regridded_data : torch.Tensor
            Regridded k-space data after Fourier and inverse Fourier transformations.
        transfer_function : torch.Tensor
            Transfer function used in the frequency domain, representing the
            encoding operator for the system.
        shape : tuple of ints
            Shape of the data in the form (nX, nY, nZ, nS, nTI, nTE), where nX, nY,
            nZ are spatial dimensions, nS is the number of channels, and nTI, nTE
            are the number of inversion and echo times, respectively.
        lambda_wavelet : float, optional
            Regularization parameter for the wavelet term to encourage sparsity in
            the wavelet domain (default is 1e-3).
        lambda_hankel : float, optional
            Regularization parameter for the Block-Hankel term to enforce structured
            low rank (default is None).
        lambda_casorati : float, optional
            Regularization parameter for the Casorati matrix to exploit local voxel
            correlations (default is None).
        weight_wavelet : float, optional
            Weight assigned to the wavelet regularizer (default is 1.0).
        weight_hankel : float, optional
            Weight assigned to the Hankel regularizer (default is 1.0).
        weight_casorati : float, optional
            Weight assigned to the Casorati regularizer (default is 1.0).
        tau : int, optional
            Step size parameter for the ADMM solver (default is 1).
        channel_normalize : bool, optional
            Whether to normalize the channel operator (default is True).
        wavelet_type : str, optional
            Type of wavelet used in the wavelet transform (default is 'db4').
        wavelet_level : int, optional
            Number of wavelet decomposition levels (default is 4).
        casorati_window : tuple of ints, optional
            Window size for forming Casorati matrices (default is (3, 3)).
        inner_iter : int, optional
            Number of inner iterations for solving the linear system using the
            conjugate gradient method (default is 5).
        outer_iter : int, optional
            Maximum number of outer iterations for the ADMM solver (default is 500).
        callback : callable, optional
            Callback function for monitoring the convergence during the optimization
            (default is None).

        Notes
        -----
        MIRAGE is part of the QRAGE framework for multiparametric quantitative MRI,
        which leverages a novel ME-MPnRAGE sequence for ultrahigh field applications,
        allowing simultaneous mapping of multiple tissue parameters. MIRAGE solves
        the corresponding inverse problem using an ADMM approach with proximal
        operators for wavelet, Hankel, and Casorati constraints.

        References
        ----------
        Zimmermann et al., QRAGE: Simultaneous multiparametric quantitative MRI of
        water content, T1, T2*, and magnetic susceptibility at ultrahigh field
        strength. Magnetic Resonance in Medicine, 2024.
        https://doi.org/10.1002/mrm.30272
        """

        num_channels = coil_sensitivities.shape[0]
        shape = regridded_data.shape

        lin_ops = []
        lin_ops_normal = []
        prox_ops = []

        channel_operator = ChannelOperator(
            coil_sensitivities,
            (num_channels,) + regridded_data.shape,
            normalize=channel_normalize,
            device=device,
        )

        transfer_function_operator = TransferFunctionOperator(
            transfer_function,
            (num_channels,) + shape,
            axes=(1, 2),
            device=device,
        )

        if lambda_wavelet is not None:
            wavelet_operator = WaveletTransformOperator(
                shape,
                axes=(0, 1),
                wavelet=wavelet_type,
                level=wavelet_level,
                device=device,
            )
            lin_ops.append(-weight_wavelet * wavelet_operator)
            lin_ops_normal.append(
                weight_wavelet**2 * IdentityOperator(shape, device=device)
            )
            prox_ops.append(
                JointSoftThresholdingOperator(
                    wavelet_operator.adjoint_shape,
                    weight_wavelet * lambda_wavelet,
                    axes=(-1, -2),
                )
            )

        if lambda_hankel is not None:
            hankel_operator = BlockHankelOperator(shape, device=device)
            lin_ops.append(-weight_hankel * hankel_operator)
            lin_ops_normal.append(
                weight_hankel**2 * BlockHankelNormalOperator(shape, device=device)
            )
            prox_ops.append(
                SingularValueSoftThresholdingOperator(
                    hankel_operator.adjoint_shape, weight_hankel * lambda_hankel
                )
            )

        if lambda_casorati is not None:
            casorati_operator = ShiftOperator(
                casorati_window, (1, 1), (0, 1), shape, device=device
            )
            lin_ops.append(-weight_casorati * casorati_operator)
            lin_ops_normal.append(
                weight_casorati**2 * IdentityOperator(shape, device=device)
            )
            prox_ops.append(
                SingularValueSoftThresholdingOperator(
                    casorati_operator.adjoint_shape,
                    weight_casorati * lambda_casorati,
                    # TODO: needs a fix
                    # (1, 2, 3, 4, 5, 6, 0),
                    # (nX, nY, nZ, nS, nTI, -1),
                )
            )

        # Define the composite operators
        self.concatenated_operator = ConcatOperator(lin_ops, device=device)
        self.identity_operator = IdentityOperator(
            self.concatenated_operator.shape[0] // 2,
            device=device,
        )
        self.constant = torch.tensor(0.0, dtype=torch.float32, device=device)

        self.concatenated_normal_operator = SumOperator(lin_ops_normal, device=device)

        # Define the proximal operators
        prox_f = SeparableProximalOperator(
            prox_ops,
            self.concatenated_operator.indices,
        )
        prox_g = LinearProximalSolver(
            regridded_data.view(torch.float32).ravel(),
            channel_operator.H @ transfer_function_operator @ channel_operator,
            self.concatenated_operator,
            self.concatenated_normal_operator,
            maxiter=cg_maxiter,
        )

        # Initialize the ADMM solver
        self.solver = ADMM(
            prox_f.solve,
            prox_g.solve,
            self.identity_operator,
            self.concatenated_operator,
            self.constant,
            maxiter=admm_maxiter,
            verbose=verbose,
            callback=callback,
            tau=tau,
            device=device,
        )

    def solve(self) -> torch.Tensor:
        """
        Solves the optimization problem using the ADMM solver.

        Returns:
        - v: The solution of the optimization problem.
        """
        return self.solver.solve()["v"]
