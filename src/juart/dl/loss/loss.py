from typing import Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from ..utils.dist import average_images
from ..utils.fourier import nonuniform_fourier_transform_forward
from ..utils.hankel import block_hankel_foward
from ..utils.validation import timing_layer, validation_layer
from ..utils.wavelet import (
    wavelet_transfer_functions,
    wavelet_transfer_functions_nd,
    wavelet_transform_forward,
)


@torch.jit.script
def svdvals(
    A: torch.Tensor, epsilon: float = 1e-10, driver: Union[str, None] = "gesvda"
) -> torch.Tensor:
    """
    Compute the singular values of a matrix A with improved numerical stability.

    To enhance the stability of the Singular Value Decomposition (SVD) calculation,
    this function adds small amounts of independent and identically distributed (IID) Gaussian noise
    to the input matrix A before computing its singular values. The noise has a standard deviation of epsilon,
    effectively regularizing the singular values. This approach is akin to adding a small constant epsilon
    to each singular value, which prevents very small or zero singular values that could lead to numerical
    instability or inaccurate gradients during backpropagation.

    Args:
        A (torch.Tensor): Input matrix for which singular values are computed.
        epsilon (float, optional): Standard deviation of the Gaussian noise added to A for regularization.
                                   Defaults to 1e-10.
        driver (Union[str, None], optional): Driver for the SVD computation. If None, the default driver is used.
                                             For CUDA computations, the default driver is 'gesvda'. Defaults to 'gesvda'.

    Returns:
        torch.Tensor: The singular values of the input matrix A.

    Notes:
        - This function is scriptable using TorchScript.
        - Adding noise to improve stability does not alter the optimal solution but stabilizes the computation.
    """
    # Add Gaussian noise to the matrix A
    A = A + torch.randn(A.shape, dtype=A.dtype, device=A.device) * epsilon

    # Concept for a deterministic approach to improve stability
    # Needs testing
    # A = A + torch.eye(A.shape[0], A.shape[1],
    #                   dtype=A.dtype,
    #                   device=A.device) * epsilon

    # Compute the singular values
    if A.is_cuda:
        S = torch.linalg.svdvals(A, driver=driver)
    else:
        S = torch.linalg.svdvals(A)

    return S


@torch.jit.script
def hankel_norm(x: torch.Tensor, indices_or_sections: int = 32) -> torch.Tensor:
    S = list()

    x = torch.flatten(x, start_dim=0, end_dim=-3)
    x = torch.tensor_split(x, indices_or_sections)

    for xi in x:
        H = block_hankel_foward(xi, (xi.shape[-2], xi.shape[-1]))
        S.append(svdvals(H))

    norm = torch.concatenate(S).sum()

    return norm


@torch.jit.script
def casorati_norm(
    x: torch.Tensor,
    kernel_size: Tuple[int, int] = (3, 3),
    padding: int = 0,
    indices_or_sections: int = 256,
) -> torch.Tensor:
    S = list()

    # Dimensions of x are nB, nX, nY, nTI, nTE
    # Dimensions should be nTI, nTE * nB, nX, nY
    x = torch.permute(x, (3, 4, 0, 1, 2))
    x = torch.flatten(x, start_dim=1, end_dim=2)

    # Dimensions of unfold are (nTI, nTE * nB * kernel_size, nN)
    # with nN number of neighborhoods
    # To get global Casorati norm, set kernel_size to (nX, nY) and padding to 0
    x = torch.nn.functional.unfold(x, kernel_size, padding=padding)

    x = x.permute((2, 1, 0))
    x = torch.tensor_split(x, indices_or_sections)

    for xi in x:
        S.append(svdvals(xi))

    norm = torch.concatenate(S).sum()

    norm /= torch.sqrt(torch.prod(torch.tensor(kernel_size)))

    return norm


def hankel_loss(x, x_reference, norm=True):
    loss = hankel_norm(x)

    if norm:
        # Upper bound of nuclear norm by sqrt(rank(H)) * frobenius norm
        H = block_hankel_foward(
            x_reference, (x_reference.shape[-2], x_reference.shape[-1])
        )
        reference_loss = torch.linalg.vector_norm(
            H, dim=(-2, -1), keepdim=True, ord=2
        ).sum()
        loss = loss / reference_loss
    else:
        loss = loss / x.numel()

    return loss


def casorati_loss(x, x_reference, kernel_size=(3, 3), padding=0, norm=True):
    loss = casorati_norm(
        x,
        kernel_size=kernel_size,
        padding=padding,
    )

    if norm:
        # Upper bound of nuclear norm by sqrt(kernel_size) * frobenius norm
        reference_loss = torch.linalg.vector_norm(
            x_reference, dim=(-2, -1), keepdim=True, ord=2
        ).sum()
        loss = loss / reference_loss
    else:
        loss = loss / x.numel()

    return loss


def absolute_loss(x, x_reference, order, dim=None, norm=True):
    loss = torch.linalg.vector_norm(
        x,
        ord=order,
        dim=dim,
        keepdim=True,
    )

    if norm:
        reference_loss = torch.linalg.vector_norm(
            x_reference,
            ord=order,
            dim=dim,
            keepdim=True,
        )
        loss = loss / reference_loss
    else:
        loss = loss / x.numel()

    return torch.mean(loss)


def difference_loss(x, x_reference, order, dim=None, norm=True):
    loss = torch.linalg.vector_norm(
        x - x_reference,
        ord=order,
        dim=dim,
        keepdim=True,
    )

    if norm:
        reference_loss = torch.linalg.vector_norm(
            x_reference,
            ord=order,
            dim=dim,
            keepdim=True,
        )
        loss = loss / reference_loss
    else:
        loss = loss / x.numel()

    return torch.mean(loss)


def centralized_loss(x, x_reference, order, dim=None, norm=True, group=None):
    x_mean = average_images(x, group=group)

    loss = torch.linalg.vector_norm(
        x - x_mean,
        ord=order,
        dim=dim,
        keepdim=True,
    )

    if norm:
        reference_loss = torch.linalg.vector_norm(
            x_reference,
            ord=order,
            dim=dim,
            keepdim=True,
        )
        loss = loss / reference_loss
    else:
        loss = loss / x.numel()

    return torch.mean(loss)


class KSpaceLoss(nn.Module):
    def __init__(
        self,
        shape,
        weights=(0.1, 0.1),
        dim=None,
        normalized_loss=True,
        timing_level=0,
        validation_level=0,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.shape = shape
        self.weights = weights
        self.dim = dim
        self.normalized_loss = normalized_loss
        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device
        self.dtype = dtype

    @timing_layer
    @validation_layer
    def forward(
        self,
        images_reconstructed,
        kspace_trajectory,
        kspace_data,
        kspace_mask,
        sensitivity_maps,
    ):
        loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        if max(self.weights) > 0:
            images_reconstructed = images_reconstructed.to(self.device)
            kspace_trajectory = kspace_trajectory.to(self.device)
            kspace_data = kspace_data.to(self.device)
            kspace_mask = kspace_mask.to(self.device)
            sensitivity_maps = sensitivity_maps.to(self.device)

            images_reconstructed = 0.5 * (
                images_reconstructed[None, ...] * sensitivity_maps[..., None, None]
            )

            kspace_data_reconstructed = nonuniform_fourier_transform_forward(
                kspace_trajectory,
                images_reconstructed,
            )

            kspace_data_reconstructed = kspace_data_reconstructed * kspace_mask
            
            kspace_data_reference = kspace_data * kspace_mask

            print(
                "[KSpaceLoss]",
                kspace_data_reconstructed.shape,
                kspace_data_reference.shape,
                self.dim,
            )

        if self.weights[0] > 0:
            loss_l1 = difference_loss(
                kspace_data_reconstructed,
                kspace_data_reference,
                order=1,
                dim=self.dim,
                norm=self.normalized_loss,
            )
            loss = loss + self.weights[0] * loss_l1

        if self.weights[1] > 0:
            loss_l2 = difference_loss(
                kspace_data_reconstructed,
                kspace_data_reference,
                order=2,
                dim=self.dim,
                norm=self.normalized_loss,
            )
            loss = loss + self.weights[1] * loss_l2

        return loss


class ImageSpaceLoss(nn.Module):
    def __init__(
        self,
        weights=(0.01, 0.01),
        dim=None,
        normalized_loss=True,
        timing_level=0,
        validation_level=0,
        group=None,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.weights = weights
        self.dim = dim
        self.normalized_loss = normalized_loss
        self.timing_level = timing_level
        self.validation_level = validation_level
        self.group = group
        self.device = device
        self.dtype = dtype

    @timing_layer
    @validation_layer
    def forward(self, images_reconstructed, images_regridded):
        loss = torch.tensor(0, device=self.device, dtype=torch.float32)

        if max(self.weights) > 0:
            images_reconstructed = images_reconstructed.to(self.device)
            images_regridded = images_regridded.to(self.device)

            print(
                "[ImageSpaceLoss]",
                images_reconstructed.shape,
                images_regridded.shape,
                self.dim,
            )

        if self.weights[0] > 0:
            loss_l1 = centralized_loss(
                images_reconstructed,
                images_regridded,
                order=1,
                dim=self.dim,
                norm=self.normalized_loss,
                group=self.group,
            )

            loss = loss + self.weights[0] * loss_l1

        if self.weights[1] > 0:
            loss_l2 = centralized_loss(
                images_reconstructed,
                images_regridded,
                order=2,
                dim=self.dim,
                norm=self.normalized_loss,
                group=self.group,
            )

            loss = loss + self.weights[1] * loss_l2

        return loss


class WaveletLoss(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 256, 256, 19, 9),
        wavelet: str = "db2",
        level: int = 4,
        axes: Tuple[int, ...] = (1, 2),
        weights=(0.0, 0.0),
        dim=None,
        normalized_loss=True,
        timing_level=0,
        validation_level=0,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        nD = (3 * level) + 1

        # Calculate wavelet transfer functions for the axes
        Hx, Gx = wavelet_transfer_functions(
            wavelet, level, input_shape[axes[0]], device=device
        )
        Hy, Gy = wavelet_transfer_functions(
            wavelet, level, input_shape[axes[1]], device=device
        )

        # Combine the transfer functions for the specified axes
        self.F = wavelet_transfer_functions_nd(
            (Hx, Hy),
            (Gx, Gy),
            (nD,) + input_shape,
            axes,
            device=device,
        )

        self.axes = axes
        self.weights = weights
        self.dim = dim
        self.normalized_loss = normalized_loss
        self.timing_level = timing_level
        self.validation_level = validation_level
        self.dtype = dtype
        self.device = device

    @timing_layer
    @validation_layer
    def forward(self, images, images_regridded):
        loss = torch.tensor(0, device=self.device, dtype=torch.float32)

        if max(self.weights) > 0:
            images = images.to(self.device)
            images_regridded = images_regridded.to(self.device)

            wavelet_coefficients_reconstruction = wavelet_transform_forward(
                images, self.F, self.axes
            )

            wavelet_coefficients_regridded = wavelet_transform_forward(
                images_regridded, self.F, self.axes
            )

            print(
                "[WaveletLoss]",
                wavelet_coefficients_reconstruction.shape,
                wavelet_coefficients_regridded.shape,
                self.dim,
            )

        if self.weights[0] > 0:
            loss_l1 = absolute_loss(
                wavelet_coefficients_reconstruction,
                wavelet_coefficients_regridded,
                order=1,
                dim=self.dim,
                norm=self.normalized_loss,
            )

            loss = loss + self.weights[0] * loss_l1

        if self.weights[1] > 0:
            loss_l2 = absolute_loss(
                wavelet_coefficients_reconstruction,
                wavelet_coefficients_regridded,
                order=2,
                dim=self.dim,
                norm=self.normalized_loss,
            )

            loss = loss + self.weights[1] * loss_l2

        return loss


class HankelLoss(nn.Module):
    def __init__(
        self,
        weights=(0, 0.001),
        normalized_loss=True,
        timing_level=0,
        validation_level=0,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.weights = weights
        self.normalized_loss = normalized_loss
        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device
        self.dtype = dtype

    @timing_layer
    @validation_layer
    def forward(self, images, images_regridded):
        images = images.to(self.device)
        images_regridded = images_regridded.to(self.device)

        loss = torch.tensor(0, device=self.device, dtype=torch.float32)

        if self.weights[0] > 0:
            loss_nuclear_diff = hankel_loss(
                torch.diff(images, dim=-2),
                torch.diff(images_regridded, dim=-2),
                norm=self.normalized_loss,
            )

            loss = loss + self.weights[0] * loss_nuclear_diff

        if self.weights[1] > 0:
            loss_nuclear = hankel_loss(
                images,
                images_regridded,
                norm=self.normalized_loss,
            )

            loss = loss + self.weights[1] * loss_nuclear

        return loss


class CasoratiLoss(nn.Module):
    def __init__(
        self,
        kernel_size,
        weights=(0, 0),
        normalized_loss=True,
        timing_level=0,
        validation_level=0,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.weights = weights
        self.normalized_loss = normalized_loss
        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device
        self.dtype = dtype

    @timing_layer
    @validation_layer
    def forward(self, images, images_regridded):
        loss = torch.tensor(0, device=self.device, dtype=torch.float32)

        if max(self.weights) > 0:
            images = images.to(self.device)
            images_regridded = images_regridded.to(self.device)

        if self.weights[0] > 0:
            kernel_size_global = (images.shape[1], images.shape[2])
            print("Kernel size (global)", kernel_size_global)

            loss_nuclear_global = casorati_loss(
                images,
                images_regridded,
                kernel_size=kernel_size_global,
                norm=self.normalized_loss,
            )

            loss = loss + self.weights[0] * loss_nuclear_global

        if self.weights[1] > 0:
            loss_nuclear = casorati_loss(
                images,
                images_regridded,
                kernel_size=self.kernel_size,
                norm=self.normalized_loss,
            )

            loss = loss + self.weights[1] * loss_nuclear

        return loss


class JointLoss(nn.Module):
    def __init__(
        self,
        shape,
        kernel_size,
        weights_kspace_loss=(0.3, 0.3),
        weights_ispace_loss=(0.15, 0.15),
        weights_wavelet_loss=(0.0, 0.0),
        weights_hankel_loss=(0.0, 0.03),
        weights_casorati_loss=(0.0, 0.0),
        dim_kspace_loss=None,
        dim_ispace_loss=None,
        dim_wavelet_loss=None,
        normalized_loss=True,
        timing_level=0,
        validation_level=0,
        group=None,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.kspace_loss = KSpaceLoss(
            shape,
            weights=weights_kspace_loss,
            dim=dim_kspace_loss,
            normalized_loss=normalized_loss,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            device=device,
            dtype=dtype,
        )
        self.ispace_loss = ImageSpaceLoss(
            weights=weights_ispace_loss,
            dim=dim_ispace_loss,
            normalized_loss=normalized_loss,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            group=group,
            device=device,
            dtype=dtype,
        )
        self.wavelet_loss = WaveletLoss(
            weights=weights_wavelet_loss,
            dim=dim_wavelet_loss,
            normalized_loss=normalized_loss,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            device=device,
            dtype=dtype,
        )
        self.hankel_loss = HankelLoss(
            weights=weights_hankel_loss,
            normalized_loss=normalized_loss,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            device=device,
            dtype=dtype,
        )
        self.casorati_loss = CasoratiLoss(
            kernel_size,
            weights=weights_casorati_loss,
            normalized_loss=normalized_loss,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            device=device,
            dtype=dtype,
        )

        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device
        self.dtype = dtype

    @timing_layer
    @validation_layer
    def forward(
        self,
        images_reconstructed,
        images_regridded,
        kspace_trajectory,
        kspace_data,
        kspace_mask,
        sensitivity_maps,
    ):
        images_reconstructed = images_reconstructed.to(self.device)
        images_regridded = images_regridded.to(self.device)
        kspace_trajectory = kspace_trajectory.to(self.device)
        kspace_data = kspace_data.to(self.device)
        kspace_mask = kspace_mask.to(self.device)
        sensitivity_maps = sensitivity_maps.to(self.device)

        loss_kspace = self.kspace_loss(
            images_reconstructed,
            kspace_trajectory,
            kspace_data,
            kspace_mask,
            sensitivity_maps,
        )

        loss_ispace = self.ispace_loss(images_reconstructed, images_regridded)

        loss_wavelet = self.wavelet_loss(images_reconstructed, images_regridded)

        loss_hankel = self.hankel_loss(images_reconstructed, images_regridded)

        loss_casorati = self.casorati_loss(images_reconstructed, images_regridded)

        print(
            (
                f"Rank {dist.get_rank()} - "
                + f"Loss k-space: {loss_kspace.cpu().detach().numpy():.3f} - "
                + f"Loss image space: {loss_ispace.cpu().detach().numpy():.3f} - "
                + f"Loss Wavelet {loss_wavelet.cpu().detach().numpy():.3f} - "
                + f"Loss Hankel {loss_hankel.cpu().detach().numpy():.3f} - "
                + f"Loss Casorati {loss_casorati.cpu().detach().numpy():.3f}"
            )
        )

        return loss_kspace + loss_ispace + loss_wavelet + loss_hankel + loss_casorati
