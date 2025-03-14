import threadpoolctl
import torch

from .pygrappa import pygrappa


def grappa(
    kspace: torch.Tensor,
    nAC: int,
    kernel_size: tuple[int, int] = (5, 5),
    lamda: float = 0.01,
    coil_axis: int = 1,
    threadpool_limits: int = 1,
    threadpool_user_api: str = "blas",
) -> torch.Tensor:
    """
    Perform GRAPPA reconstruction on k-space data.

    Parameters
    ----------
    kspace : torch.Tensor
        The input k-space data.
    nAC : int
        Number of autocalibration lines.
    kernel_size : tuple[int, int]
        The GRAPPA kernel size.
    lamda : float
        Regularization parameter for GRAPPA.
    coil_axis : int
        The axis representing the coils.
    threadpool_limits : int
        Limit the number of threads for the BLAS library.
    threadpool_user_api : str
        The API for threadpool control (e.g., 'blas').

    Returns
    -------
    torch.Tensor
        The reconstructed k-space data.
    """
    # Convert PyTorch tensor to NumPy array for pygrappa
    kspace_np = kspace.cpu().numpy()

    # Extract autocalibration area
    calib = kspace_np[
        :, :, (kspace_np.shape[2] - nAC) // 2 : (kspace_np.shape[2] + nAC) // 2
    ]

    # Perform GRAPPA reconstruction
    with threadpoolctl.threadpool_limits(
        limits=threadpool_limits, user_api=threadpool_user_api
    ):
        kspace_grappa_np = pygrappa(
            kspace_np, calib, coil_axis=coil_axis, kernel_size=kernel_size, lamda=lamda
        )

    # Convert the result back to PyTorch tensor
    kspace_grappa = torch.tensor(
        kspace_grappa_np, dtype=kspace.dtype, device=kspace.device
    )

    return kspace_grappa
