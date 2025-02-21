from typing import Tuple

import torch


def ac_region(
    kspace_trajectory: torch.Tensor,
    kspace_data: torch.Tensor,
    calibration_region_size: int,
    kspace_size: int,
    ord: float = torch.inf,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the autocalibration (AC) region from k-space data and resize it.

    Parameters
    ----------
    kspace_trajectory : torch.Tensor
        K-space trajectory of shape (..., M, N).
    kspace_data : torch.Tensor
        Corresponding k-space data of shape (..., M, N, ...).
    calibration_region_size : int
        Size of the calibration region in k-space.
    kspace_size : int
        Full k-space size.
    ord : float, optional
        Order of the norm for selecting the AC region (default: infinity norm).

    Returns
    -------
    torch.Tensor
        Rescaled k-space trajectory in the AC region.
    torch.Tensor
        Rescaled k-space data in the AC region.
    """

    # Compute the Euclidean norm along the k-space trajectory dimensions
    trajectory_norm = torch.linalg.vector_norm(
        kspace_trajectory[..., 0, :, :], ord=ord, dim=0
    )

    # Identify indices inside the calibration region
    ac_mask = trajectory_norm <= (0.5 * calibration_region_size / kspace_size)
    index = torch.nonzero(ac_mask, as_tuple=True)[0]  # Extract indices

    # Extract and rescale k-space trajectory
    kspace_trajectory_ac = kspace_trajectory[:, index, :, :, :] * (
        kspace_size / calibration_region_size
    )

    # Extract and rescale k-space data
    kspace_data_ac = kspace_data[:, :, :, index, :, :, :] * (
        calibration_region_size / kspace_size
    )

    return kspace_trajectory_ac, kspace_data_ac
