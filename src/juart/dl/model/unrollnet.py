import itertools
from typing import Tuple
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from ..utils.validation import timing_layer, validation_layer
from .dc import DataConsistency
from .resnet import ResNet
from .unet import UNet



class ExponentialMovingAverageModel(AveragedModel):
    def __init__(
        self,
        model,
        decay=0.9,
    ):
        super().__init__(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(decay),
        )


class LookaheadModel(AveragedModel):
    def __init__(
        self,
        model,
        alpha=0.5,
        k=5,
    ):
        super().__init__(
            model,
        )

        self.alpha = alpha
        self.k = k
        self.step_counter = 0

    def update_parameters(
        self,
        model,
    ):
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            print(f"Rank {dist.get_rank()} - Updating Slow and Fast Weights ...")
            for p_slow, p_fast in zip(self.module.parameters(), model.parameters()):
                if p_fast.requires_grad:
                    p_slow.data.add_(p_fast.data - p_slow.data, alpha=self.alpha)
                    p_fast.data.copy_(p_slow.data)


class UnrolledNet(nn.Module):
    def __init__(
        self,
        shape,
        CG_Iter=10,
        num_unroll_blocks=10,
        num_res_blocks=15,
        features=128,
        activation="ReLU",
        lamda_start=0.05,
        phase_normalization=False,
        disable_progress_bar=False,
        timing_level=0,
        validation_level=0,
        kernel_size: Tuple[int] = (3, 3),
        regularizer="ResNet",
        axes: Tuple[int] = (1, 2),
        device=None,
        ConvLayerCheckpoints: bool = False,
        ResNetCheckpoints: bool = False,
        dtype=torch.complex64,
    ):
        """
        Initializes an UnrollNet as a neural Network with a number of ResNet Layers (ConvLayers) and a data consistency layer.

        Parameters
        ----------
        shape : torch.Tensor, shape (nX, nY, nZ, nTI, nTE)
            Shape of the image data.
        CG_Iter : int, optional
            Number of the iterations in the conjugate gradient (Data Consistency) term
        num_unroll_blocks : int, optional
            Number of iterations in the loop of data consistency term and regularization
            term (default is 10).
        num_res_blocks : int, optional
            Number of ResNetBlocks that should be added to the second layer of the
            ResNet (default is 15).
        features : int, optional
            Number of the features of the neural network (default is 128).
        weight_standardization : bool, optional
            Activates the weight standardization that sets the mean of the weights to 0
            and their deviation to 1(default is False).
        spectral_normalization: bool, optional
            Activates the spectral normalization (default is False).
        phase_normalization: bool, optional
            normalizes the signals phase (default is False).
        disable_progress_bar: bool, optional
            Disable the progress bar output (default is False).
        axes: tuple[int], optional
            Defines the dimension of the model (default is (1,2) for 2D; Change to
            (1,2,3) for 3D)
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        kernel_size: Tuple[int], optional
            changes the size of the kernel used in the convolutional layers
            (default is (3,3))
        device : str, optional
            Device on which to perform the computation
            (default is None, which uses the current device).
            It is also possible to give a list of strings. The first
            item is the DataConsistency device and the second one is
            the device used for the resnet.

        NOTE: This function is under development and may not be fully functional yet.
        """
        super().__init__()

        self.phase_normalization = phase_normalization
        self.num_unroll_blocks = num_unroll_blocks
        self.disable_progress_bar = disable_progress_bar
        self.timing_level = timing_level
        self.validation_level = validation_level
        
        nX, nY, nZ, nTI, nTE = shape
        contrasts = nTI * nTE
        dim = len(axes)

        if type(device) == list:
            
            if len(device) > 1:
                dc_device = device[0]
                resnet_device = device[1]
                
            else:
                dc_device = device[0]
                resnet_device = device[0]

        else:
            dc_device = device
            resnet_device = device

        if regularizer == "ResNet":
            self.regularizer = ResNet(
                contrasts=contrasts,
                features=features,
                num_of_resblocks=num_res_blocks,
                activation=activation,
                kernel_size=kernel_size,
                ResNetCheckpoints = ResNetCheckpoints,
                timing_level=timing_level - 1,
                validation_level=validation_level - 1,
                dim=dim,
                device=device,
                dtype=dtype,
            )

        elif regularizer == "UNet":
            self.regularizer = UNet(
                contrasts=contrasts,
                features=features,
                device=device,
                dtype=dtype,
            )

        self.dc = DataConsistency(
            shape,
            niter=CG_Iter,
            lamda_start=lamda_start,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            axes=axes,
            device=device,
            dtype=dtype,
        ).to(device)

    @timing_layer
    @validation_layer
    def forward(
        self,
        images_regridded: torch.Tensor,
        kspace_trajectory: torch.Tensor,
        kspace_mask: torch.Tensor = None,
        sensitivity_maps: torch.Tensor = None,
    ) -> torch.Tensor:
        
        if self.phase_normalization:
            images_phase = torch.exp(1j * torch.angle(images_regridded[..., 0, 0]))
            images_regridded = images_regridded / images_phase[..., None, None]
            sensitivity_maps = sensitivity_maps * images_phase[None, :, :]

        self.dc.init(
            images_regridded,
            kspace_trajectory,
            kspace_mask=kspace_mask,
            sensitivity_maps=sensitivity_maps,
        )

        images = images_regridded.clone().detach()

        for _ in tqdm(range(self.num_unroll_blocks), disable=self.disable_progress_bar):
            images = checkpoint(self.regularizer, images, use_reentrant=False)
            images = checkpoint(self.dc, images, use_reentrant=False)

        if self.phase_normalization:
            images = images * images_phase[..., None, None]

        return images


class SingleContrastUnrolledNet(nn.Module):
    def __init__(
        self,
        shape,
        CG_Iter=10,
        num_unroll_blocks=10,
        num_res_blocks=15,
        contrasts=1,
        features=32,
        weight_standardization=False,
        spectral_normalization=False,
        activation="ReLU",
        lamda_start=0.05,
        phase_normalization=False,
        disable_progress_bar=False,
        timing_level=0,
        validation_level=0,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        nX, nY, nZ, nTI, nTE = shape
        contrasts = nTI * nTE

        self.unrollednets = nn.ModuleList(
            [
                UnrolledNet(
                    (nX, nY, nZ, 1, 1),
                    CG_Iter=CG_Iter,
                    num_unroll_blocks=num_unroll_blocks,
                    num_res_blocks=num_res_blocks,
                    features=features,
                    weight_standardization=weight_standardization,
                    spectral_normalization=spectral_normalization,
                    activation=activation,
                    lamda_start=lamda_start,
                    phase_normalization=phase_normalization,
                    disable_progress_bar=True,
                    timing_level=timing_level - 1,
                    validation_level=validation_level - 1,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(contrasts)
            ]
        )

        self.disable_progress_bar = disable_progress_bar
        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device

    @timing_layer
    @validation_layer
    def forward(
        self,
        images_regridded: torch.Tensor,
        kspace_trajectory: torch.Tensor,
        kspace_mask: torch.Tensor = None,
        sensitivity_maps: torch.Tensor = None,
    ) -> torch.Tensor:
        nTI, nTE = images_regridded.shape[-2:]

        nX, nY, nZ, nTI, nTE = images_regridded.shape

        images = images_regridded.clone().detach()

        for iTI, iTE in tqdm(
            itertools.product(range(nTI), range(nTE)),
            total=nTI * nTE,
            disable=self.disable_progress_bar,
        ):
            if kspace_mask is not None:
                images[..., iTI : iTI + 1, iTE : iTE + 1] = self.unrollednets[
                    iTI * nTE + iTE
                ](
                    images_regridded[..., iTI : iTI + 1, iTE : iTE + 1],
                    kspace_trajectory[..., iTI : iTI + 1, iTE : iTE + 1],
                    kspace_mask=kspace_mask[..., iTI : iTI + 1, iTE : iTE + 1],
                    sensitivity_maps=sensitivity_maps,
                )
            else:
                images[..., iTI : iTI + 1, iTE : iTE + 1] = self.unrollednets[
                    iTI * nTE + iTE
                ](
                    images_regridded[..., iTI : iTI + 1, iTE : iTE + 1],
                    kspace_trajectory[..., iTI : iTI + 1, iTE : iTE + 1],
                    kspace_mask=None,
                    sensitivity_maps=sensitivity_maps,
                )
        return images
