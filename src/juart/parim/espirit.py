import itertools
import logging
from typing import Tuple

import torch

from ..conopt.functional import crop_tensor, pad_tensor
from ..conopt.functional.fourier import fourier_transform_forward
from ..conopt.functional.system import system_matrix_forward

logger = logging.getLogger(__name__)


def espirit(
    kspace_data: torch.Tensor,
    image_shape: Tuple[int, ...],
    calibration_region_size: Tuple[int, ...] = (24, 24, 24),
    kernel_size: Tuple[int, ...] = (6, 6, 6),
    eigenvalue_threshold_1: int = 0.02,
    eigenvalue_threshold_2: int = 0.95,
    num_singular_values: int = 1,
    threshold_mode: str = "hard",
) -> torch.Tensor:
    """
    ESPIRiT Algorithm to estimate coil sensitivity maps.

    Parameters
    ----------
    kspace_data : nd-array
        The input k-space data.
    image_shape : tuple of ints
        The shape of the target image.
    calibration_region_size : tuple of ints, optional
        Size of the calibration region.
    kernel_size : int or tuple of ints, optional
        Size of the kernel.
    eigenvalue_threshold_1 : float, optional
        Threshold for selecting singular vectors of the calibration matrix.
    eigenvalue_threshold_2 : float, optional
        Threshold for eigenvector decomposition in image space.
    num_singular_values : int, optional
        Number of singular values.
    threshold_mode : string, optional
        Eigenvalue maps are hard- or soft-thresholded.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    sensitivity_maps : nd-array
        Sensitivity maps.

    Notes
    -----
    .. [1] M. Uecker et al., “ESPIRiT - An eigenvalue approach to
           autocalibrating parallel MRI: Where SENSE meets GRAPPA,” Magnetic
           Resonance in Medicine, vol. 71, no. 3, pp. 990–1001, 2014.
    """

    logger.info("Starting ESPIRiT Coil Sensitivity Estimation ...")

    num_channels, num_kspace_x, num_kspace_y, num_kspace_z, num_contrasts = (
        kspace_data.shape
    )

    # Calibration region cannot be larger than kspace data
    calibration_region_size = (
        min(calibration_region_size[0], num_kspace_x),
        min(calibration_region_size[1], num_kspace_y),
        min(calibration_region_size[2], num_kspace_z),
    )

    # Kernel cannot be larger than calibration region
    kernel_size = (
        min(kernel_size[0], calibration_region_size[0]),
        min(kernel_size[1], calibration_region_size[1]),
        min(kernel_size[2], calibration_region_size[2]),
    )

    # Crop calibration area
    calibration_data = crop_tensor(
        kspace_data, (num_channels,) + calibration_region_size + (num_contrasts,)
    )

    kspace_kernel, singular_values = compute_kspace_kernel(
        calibration_data, kernel_size
    )

    # Compute eigenvalue decomposition to get sensitivity maps
    eigenvectors, eigenvalues = compute_eigen_decomposition(
        kspace_kernel, singular_values, kernel_size, image_shape, eigenvalue_threshold_1
    )

    sensitivity_maps = compute_sensitivity_maps(
        eigenvectors,
        eigenvalues,
        eigenvalue_threshold_2,
        num_singular_values,
        threshold_mode,
    )

    sensitivity_maps = sensitivity_maps.permute((4, 3, 0, 1, 2))

    if num_singular_values == 1:
        sensitivity_maps = sensitivity_maps[0, ...]

    logger.info("Completed ESPIRiT Coil Sensitivity Estimation.")

    return sensitivity_maps


def compute_kspace_kernel(
    kspace_data: torch.Tensor,
    kernel_size: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform k-space calibration step for ESPIRiT and create k-space kernels.
    """

    logger.info("Computing k-space kernel: 0% complete")

    kspace_data = kspace_data[..., None, None]
    calibration_matrix = system_matrix_forward(
        kspace_data, kspace_data.shape, kernel_size
    )
    calibration_matrix *= torch.sqrt(torch.prod(torch.tensor(kernel_size)))
    calibration_matrix = calibration_matrix[..., 0, 0]

    calibration_matrix = calibration_matrix.permute((3, 2, 1, 0))
    num_contrasts, num_elements, num_shifts, num_channels = calibration_matrix.shape
    calibration_matrix = torch.reshape(
        calibration_matrix, (num_contrasts * num_elements, num_shifts * num_channels)
    )

    logger.debug(
        f"num_contrasts: {num_contrasts}, num_elements: {num_elements}, "
        f"num_shifts: {num_shifts}, num_channels: {num_channels}"
    )
    logger.info("Computing k-space kernel: 50% complete")

    U, S, V = torch.linalg.svd(calibration_matrix, full_matrices=True)
    kspace_kernel = (
        torch.conj(V)
        .permute((1, 0))
        .reshape(kernel_size + (num_channels, num_shifts * num_channels))
    )

    logger.debug(f"U.shape: {U.shape}, S.shape: {S.shape}, V.shape: {V.shape}")
    logger.debug(f"kspace_kernel.shape: {kspace_kernel.shape}")
    logger.info("Computing k-space kernel: 100% complete")

    return kspace_kernel, S


def compute_eigen_decomposition(
    kspace_kernel: torch.Tensor,
    singular_values: torch.Tensor,
    kernel_size: Tuple[int, ...],
    image_shape: Tuple[int, ...],
    eigenvalue_threshold_1: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the eigenvalue decomposition of a k-space kernel in image space.
    """

    logger.info("Computing eigendecomposition: 0% complete")

    index = torch.where(singular_values >= singular_values[0] * eigenvalue_threshold_1)[
        0
    ].max()
    kspace_kernel = kspace_kernel[..., : index + 1]

    logger.debug(f"index: {index}, kspace_kernel.shape: {kspace_kernel.shape}")
    logger.info("Computing eigendecomposition: 30% complete")

    num_channels, num_vectors = kspace_kernel.shape[-2:]

    reshaped_kernel = kspace_kernel.swapaxes(-1, -2)
    reshaped_kernel = reshaped_kernel.reshape(
        (torch.prod(torch.tensor(kernel_size)) * num_vectors, num_channels)
    )

    if reshaped_kernel.shape[0] < reshaped_kernel.shape[1]:
        u, s, v = torch.linalg.svd(reshaped_kernel)
    else:
        u, s, v = torch.linalg.svd(reshaped_kernel, full_matrices=False)

    reshaped_kernel = reshaped_kernel @ (torch.conj(v).T)

    kspace_kernel = reshaped_kernel.reshape(kernel_size + (num_vectors, num_channels))
    kspace_kernel = kspace_kernel.swapaxes(-1, -2)

    kspace_kernel = pad_tensor(kspace_kernel, image_shape + kspace_kernel.shape[-2:])
    kspace_kernel *= torch.sqrt(
        torch.prod(torch.tensor(image_shape)) / torch.prod(torch.tensor(kernel_size))
    )

    # Perform Fourier Transform on the kernels
    for channel, vector in itertools.product(range(num_channels), range(num_vectors)):
        kspace_kernel[..., channel, vector] = fourier_transform_forward(
            kspace_kernel[..., channel, vector], (0, 1, 2)
        )

    logger.debug(f"kspace_kernel.shape: {kspace_kernel.shape}")
    logger.info("Computing eigendecomposition: 60% complete")

    eigenvectors, eigenvalues, _ = torch.linalg.svd(kspace_kernel, full_matrices=False)

    # Phase correction of eigenvectors
    eigenvectors *= torch.exp(-1j * torch.angle(eigenvectors[..., :1, :]))
    eigenvectors = -torch.conj(eigenvectors)

    eigenvectors = torch.matmul(torch.conj(v).T, eigenvectors)

    logger.debug(
        f"eigenvectors.shape: {eigenvectors.shape}, "
        f"eigenvalues.shape: {eigenvalues.shape}"
    )
    logger.info("Computing eigendecomposition: 100% complete")

    return eigenvectors, eigenvalues


def compute_sensitivity_maps(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvalue_threshold_2: float,
    num_singular_values: int = 1,
    threshold_mode: str = "hard",
) -> torch.Tensor:
    """
    Generate sensitivity maps from eigenvectors and eigenvalues.
    """

    eigenvalues = eigenvalues[..., None, :]

    if threshold_mode == "hard":
        eigenvalues = eigenvalues > eigenvalue_threshold_2
    elif threshold_mode == "soft":
        eigenvalues = (
            (eigenvalues - eigenvalue_threshold_2)
            / (1 - eigenvalue_threshold_2)
            * (eigenvalues > eigenvalue_threshold_2)
        )
        eigenvalues = (1 - torch.cos(torch.pi * eigenvalues)) / 2
        eigenvalues = torch.sqrt(eigenvalues)
    else:
        raise ValueError("Invalid threshold_mode. Choose 'hard' or 'soft'.")

    sensitivity_maps = (
        eigenvectors[..., :num_singular_values] * eigenvalues[..., :num_singular_values]
    )

    return sensitivity_maps
