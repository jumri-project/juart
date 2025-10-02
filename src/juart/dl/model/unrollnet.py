import itertools
from typing import Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from ..utils.validation import timing_layer, validation_layer
from .dc import DataConsistency
from .regularizer import Regularizer


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
        num_of_resblocks=15,
        features=32,
        activation="ReLU",
        lamda_start=0.05,
        phase_normalization=False,
        disable_progress_bar=False,
        pad_to: int = 0,
        timing_level=0,
        validation_level=0,
        kernel_size: Tuple[int] = (3, 3),
        regularizer="ResNet",
        device=None,
        Checkpoints:bool = False,
        dtype=torch.complex64,
    ):
        """
        Initializes an UnrollNet as a neural Network existing out of a regularizer
        and a data consistency layer.

        Parameters
        ----------
        shape : torch.Tensor, shape (nX, nY, nZ, nTI, nTE)
            Shape of the image data.
        CG_Iter : int, optional
            Number of the iterations in the conjugate gradient (Data Consistency) term
        num_unroll_blocks : int, optional
            Number of iterations in the loop of data consistency term and regularization
            term (default is 10).
        num_of_res_blocks : int, optional
            Number of ResNetBlocks that should be added to the second layer of the
            ResNet (default is 15).
        features : int, optional
            Number of the features of the neural network (default is 128).
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        phase_normalization: bool, optional
            normalizes the signals phase (default is False).
        disable_progress_bar: bool, optional
            Disable the progress bar output (default is False).
        kernel_size: Tuple[int], optional
            changes the size of the kernel used in the convolutional layers and  its length
            decides whether all operations should be 2D or 3D. (default is (3,3))
        regularizer: str, optional
            decides which regularizer should be used. For now there are ResNet and UNet
            (default is ResNet).
        pad_to: int, optional
            provides the ability to pad the input image to the shape (pad_to,pad_to,1) or
            (pad_to,pad_to,pad_to) depending on the length of the kernel_size. Originally
            used for the UNet and its dependency on the shape of 2^n. If pad_to = 0 and
            UNet is used than the shape will be padded to the next 2^n shape.
            (default is 0)
        device : str, optional
            Device on which to perform the computation
            (default is None, which uses the current device).
            It is also possible to give a list of strings. The first
            item is the DataConsistency device and the second one is
            the device used for the regularizer.
        Checkpoints: bool, optional
            If true then checkpoints will be added in the regularizer, providing lower memory
            usage to the cost of higher computing time (default is False).

        NOTE: This function is under development and may not be fully functional yet.
        """
        super().__init__()

        axes = ([n for n in range(1, len(kernel_size)+1, 1)])
        self.pad_to = pad_to
        self.net_structure = regularizer
        self.kernel_size = kernel_size
        self.phase_normalization = phase_normalization
        self.num_unroll_blocks = num_unroll_blocks
        self.disable_progress_bar = disable_progress_bar
        self.timing_level = timing_level
        self.validation_level = validation_level

        nX, nY, nZ, nTI, nTE = shape
        contrasts = nTI * nTE

        if type(device) == list:

            if len(device) > 1:
                dc_device = device[0]
                resnet_device = device[1]

            else:
                dc_device = device[0]
                resnet_device = device[0]

        else:
            dc_device = device
            reg_device = device

        self.regularizer = Regularizer(
            shape,
            regularizer=regularizer,
            features=features,
            activation=activation,
            kernel_size=kernel_size,
            num_of_resblocks=num_of_resblocks,
            Checkpoints=Checkpoints,
            timing_level=timing_level,
            validation_level=validation_level,
            device=reg_device,
            dtype=dtype
        )

        if regularizer == "UNet":
            corr = int(2**torch.ceil(torch.log2(torch.Tensor([shape[0]]))).item())
            shape = (corr, corr, corr, shape[3], shape[4])

            if pad_to != 0:
                if len(kernel_size) == 2:
                    shape = (pad_to,pad_to,shape[2],shape[3],shape[4])

                elif len(kernel_size) == 3:
                    shape = (pad_to,pad_to,pad_to,shape[3],shape[4])

        self.dc = DataConsistency(
            shape,
            niter=CG_Iter,
            lamda_start=lamda_start,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
            axes=axes,
            device=dc_device,
            dtype=dtype,
        )

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

        image = images_regridded.clone().detach()

        for _ in tqdm(range(self.num_unroll_blocks), disable=self.disable_progress_bar):
            image = checkpoint(self.regularizer, image, use_reentrant=False)
            image = checkpoint(self.dc, image, use_reentrant=False)

        if self.phase_normalization:
            image = image * images_phase[..., None, None]

        return image


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
