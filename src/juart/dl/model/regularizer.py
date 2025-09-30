import torch
from torch import nn

from resnet import ResNet
from unet import UNet

class Regularizer(nn.Module):
    def __init__(
        shape: tuple[int],
        regularizer: str = "ResNet",
        features: list = [32],
        activation: str = "ReLU",
        kernel_size: tuple[int] = (3,3),
        num_of_resblocks: int = 10,
        Checkpoints: bool = False,
        timing_level: int = 0,
        validation_level: int = 0,
        device: str = None,
        dtype = torch.complex64
    ):

    _, _, _, nTI, nTE = shape
    contrasts = nTI*nTE
    super().__init__()
    
    if regularizer == "ResNet":
        self.regularizer = ResNet(
            contrasts=contrasts,
            features=features,
            num_of_resblocks=num_of_resblocks,
            activation=activation,
            kernel_size=kernel_size,
            ResNetCheckpoints = Checkpoints,
            timing_level=timing_level - 1,
            validation_level=validation_level - 1,
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


    @timing_layer
    @validation_layer
    def forward(
        self,
        images_regridded: torch.Tensor,
        kspace_trajectory: torch.Tensor,
        kspace_mask: torch.Tensor = None,
        sensitivity_maps: torch.Tensor = None,
    ) -> torch.Tensor:
        
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
            images = images * images_phase[..., None, None]

        return image