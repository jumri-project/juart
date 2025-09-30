from typing import Tuple

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from ..utils.validation import timing_layer, validation_layer
from common import ComplexActivation, ConvLayer, DoubleConv


class ResNetBlocksModule(nn.Module):
    def __init__(
        self,
        features,
        kernel_size=(3, 3),
        activation="ReLU",
        num_of_resblocks=15,
        scale_factor=0,
        device=None,
        ConvLayerCheckpoints = False,
        ResNetCheckpoints = False,
        dtype=torch.complex64,
    ):
        """
        Initializes a ResNetBlocksModule class that calls a variable number of
        ResNetBlocks and can let them calculate in its forward pass.

        Parameters
        ----------
        features : int
            number of the features of the ResNet. The more features it has the more
            things it can learn.
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions (X, Y, Z)
            (default is (3,3)).
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        num_of_resblocks : int, optional
            number of ResNetBlocks that should be initialized. Everyone of them
            containing 2 ConvLayers (default is 15).
        scale_factor: float, optional
            defines the impact of the ResNetBlocks on the image (default is 0.1)
        kernel_size: Tuple[int], optional
            changes the size of the kernel used in the convolutional layers
            (default is (3,3))
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).

        NOTE: This function is under development and may not be fully functional yet.
        """
        super().__init__()
        self.ResNetCheckpoints = ResNetCheckpoints
        self.layers = nn.ModuleList(
            [
                DoubleConv(
                    features,
                    kernel_size,
                    activation=activation,
                    device=device,
                    ConvLayerCheckpoints = ConvLayerCheckpoints,
                    dtype=dtype,
                )
                for _ in range(num_of_resblocks)
            ]
        )

        self.scale_factor = scale_factor
        self.device = device

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        if self.ResNetCheckpoints:
            for i, layer in enumerate(self.layers):
                images = checkpoint(self.CalcLayer, images, layer, use_reentrant=False)

        else:
            for i, layer in enumerate(self.layers):
                images = images + self.scale_factor * layer(images)

        return images

    def CalcLayer(
        self,
        images: torch.Tensor,
        layer,
    ) -> torch.Tensor:

        return images + self.scale_factor * layer(images)


class ResNet(nn.Module):
    def __init__(
        self,
        contrasts=1,
        features=128,
        num_of_resblocks=15,
        kernel_size: Tuple[int] = (3, 3),
        activation="ReLU",
        timing_level=0,
        validation_level=0,
        device=None,
        ConvLayerCheckpoints: bool = False,
        ResNetCheckpoints = False,
        dtype=torch.complex64,
    ):
        """
        Initializes a ResNet class with a variable number of resblocks and ConvLayers.

        Parameters
        ----------
        contrast : int, optional
            number of contrasts that should be respected (default is 1).
        features : int, optional
            number of the features of the ResNet. The more features it has the more
            things it can learn (default is 128).
        num_of_resblocks : int, optional
            number of ResNetBlocks that should be initialized. Everyone of them
            containing 2 ConvLayers (default is 15).
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions (X, Y, Z)
            (default is (3,3)).
        activation: str, optional
            defines the kind of activation function (default is "ReLu")
        device : torch.device, optional
            Device on which to perform the computation
            (default is None, which uses the current device).

        NOTE: This function is under development and may not be fully functional yet.
        """
        super().__init__()

        self.layer1 = ConvLayer(
            contrasts,
            contrasts,
            kernel_size,
            activation="Identity",
            device=device,
            ConvLayerCheckpoints = ConvLayerCheckpoints,
            dtype=dtype,
        )

        self.layer2 = ResNetBlocksModule(
            features,
            kernel_size,
            activation=activation,
            num_of_resblocks=num_of_resblocks,
            device=device,
            ConvLayerCheckpoints = ConvLayerCheckpoints,
            ResNetCheckpoints = ResNetCheckpoints,
            dtype=dtype,
        )

        self.layer3 = ConvLayer(
            features,
            features,
            kernel_size,
            activation="Identity",
            device=device,
            ConvLayerCheckpoints = ConvLayerCheckpoints,
            dtype=dtype,
        )

        self.layer4 = ConvLayer(
            features,
            contrasts,
            kernel_size,
            activation="Identity",
            device=device,
            ConvLayerCheckpoints = ConvLayerCheckpoints,
            dtype=dtype,
        )

        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device

    @timing_layer
    @validation_layer
    def forward(
        self,
        images: torch.Tensor,  # shape: [nX,nY,nZ,nTI,nTE]
    ) -> torch.Tensor:
        images = images.to(self.device)

        if len(kernel_size) == 2:
            nTI, nTE = images.shape[-2:]

            images = images[
                None, :, :, 0, :, :
            ]  # switches shape to [blank, nX, nY, nTI, nTE]
            images = torch.permute(
                images, (0, 3, 4, 1, 2)
            )  # switches shape to [blank, nTI, nTE, nX, nY]
            images = torch.flatten(
                images, start_dim=1, end_dim=2
            )  # switches shape to [blank, nTI*nTE, nX, nY]

            l1_out = self.layer1(images)
            l2_out = self.layer2(l1_out)
            l3_out = self.layer3(l2_out)
            images = self.layer4(l3_out + l1_out)

            images = torch.unflatten(
                images, 1, (nTI, nTE)
            )  # switches shape to [blank, nTI, nTE, nX, nY]
            images = torch.permute(
                images, (0, 3, 4, 1, 2)
            )  # switches shape to [blank, nX, nY, nTI, nTE]
            images = images[
                0, :, :, None, :, :
            ]  # switches shape back to [nX, nY, nZ, nTI, nTE]

        if len(kernel_size) == 3:
            nTI, nTE = images.shape[-2:]

            images = torch.permute(
                images, (3, 4, 0, 1, 2)
            )  # switches shape to [nTI, nTE, nX, nY, nZ]
            images = torch.flatten(
                images, start_dim=0, end_dim=1
            )  # switches shape to [nTI * nTE, nX, nY, nZ]

            images = self.layer1(images)
            #l2_out = self.layer2(l1_out)
            #l3_out = self.layer3(l2_out)
            #images = self.layer4(l3_out + l1_out)

            images = torch.unflatten(
                images, 0, (nTI, nTE)
            )  # switches shape to [nTI, nTE, nX, nY, nZ]
            images = torch.permute(
                images, (2, 3, 4, 0, 1)
            )  # switches shape to [nX, nY, nZ, nTI, nTE]

        return images
