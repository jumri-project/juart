import itertools

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from ..utils.validation import timing_layer, validation_layer
from .dc import DataConsistency
from .resnet import ResNet


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
        im_size,
        grid_size=None,
        CG_Iter=10,
        num_unroll_blocks=10,
        num_res_blocks=15,
        contrasts=1,
        features=128,
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

        self.regularizer = ResNet(
            contrasts=contrasts,
            features=features,
            num_of_resblocks=num_res_blocks,
            weight_standardization=weight_standardization,
            spectral_normalization=spectral_normalization,
            activation=activation,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            device=device,
            dtype=dtype,
        )
        self.dc = DataConsistency(
            im_size,
            grid_size=grid_size,
            niter=CG_Iter,
            lamda_start=lamda_start,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            device=device,
            dtype=dtype,
        )
        self.num_unroll_blocks = num_unroll_blocks
        self.phase_normalization = phase_normalization
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
        im_size,
        grid_size=None,
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

        self.unrollednets = nn.ModuleList(
            [
                UnrolledNet(
                    im_size,
                    grid_size=grid_size,
                    CG_Iter=CG_Iter,
                    num_unroll_blocks=num_unroll_blocks,
                    num_res_blocks=num_res_blocks,
                    contrasts=1,
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
