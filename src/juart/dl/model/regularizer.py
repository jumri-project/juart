import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from ..utils.validation import timing_layer, validation_layer

from .resnet import ResNet
from .unet import UNet


class Regularizer(nn.Module):
    def __init__(
        self,
        shape: tuple[int],
        regularizer: str = "ResNet",
        features: list = [32],
        activation: str = "ReLU",
        kernel_size: tuple[int] = (3, 3),
        num_of_resblocks: int = 10,
        Checkpoints: bool = False,
        timing_level: int = 0,
        validation_level: int = 0,
        device: str = None,
        dtype=torch.complex64
    ):

        _, _, _, nTI, nTE = shape
        contrasts = nTI*nTE
        super().__init__()

        self.timing_level = timing_level
        self.validation_level = validation_level
        self.kernel_size = kernel_size

        if regularizer == "ResNet":
            self.regularizer = ResNet(
                contrasts=contrasts,
                features=features,
                num_of_resblocks=num_of_resblocks,
                activation=activation,
                kernel_size=kernel_size,
                ResNetCheckpoints=Checkpoints,
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
    ) -> torch.Tensor:

        image = images_regridded.clone().detach()

        nTI, nTE = image.shape[-2:]

        image = image.flatten(start_dim=3, end_dim=4)  # [nX,nY,nZ,nTI*nTE]
        image = image.permute((3, 0, 1, 2))  # [nTI*nTE,nX,nY,nZ]
        image = image[None, :, :, :, :]  # [blank,nTI*nTE,nX,nY,nZ]

        if len(self.kernel_size) == 2:
            if image.shape[-1] == 1:
                image = image[:, :, :, :, 0]  # [blank,nTI*nTE,nX,nY]

            else:
                raise ValueError('z-dimension will be killed when using a 2D kernel on a 3D dataset')

        image = self.regularizer(image)

        if len(self.kernel_size) == 2:
            image = image[:, :, :, :, None]  # [blank,nTI*nTE,nX,nY,nZ]

        image = image[0, :, :, :, :]
        image = image.unflatten(0, (nTI, nTE))  # switches shape to [blank,nTI,nTE,nX,nY,nZ]
        image = image.permute((2, 3, 4, 0, 1))  # switches shape to [nX,nY,nZ,nTI,nTE]

        return image