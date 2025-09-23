import torch
import torch.nn as nn


class ComplexActivation(nn.Module):
    def __init__(
        self,
        activation: str = "ReLU",
    ):
        super().__init__()

        if activation == "ELU":
            self.functional = nn.functional.elu

        elif activation == "LeakyReLU":
            self.functional = nn.functional.leaky_relu

        elif activation == "ReLU":
            self.functional = nn.functional.relu

        else:
            raise ValueError(f"Unsupported activation_type: {activation}.")

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return torch.complex(
            self.functional(images.real),
            self.functional(images.imag),
        )


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        padding=1,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                device=device,
                dtype=dtype,
            ),
            ComplexActivation(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                device=device,
                dtype=dtype,
            ),
            ComplexActivation(),
        )

    def forward(
        self,
        images,
    ):
        return self.double_conv(images)


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            device=device,
            dtype=dtype,
        )
        self.conv = DoubleConv(
            out_channels,
            out_channels,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        images,
    ):
        images = self.down(images)
        images = self.conv(images)
        return images


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            device=device,
            dtype=dtype,
        )
        self.conv = DoubleConv(
            out_channels * 2,
            out_channels,
            device=device,
            dtype=dtype,
        )  # *2 for skip connection

    def forward(
        self,
        images,
        skip,
    ):
        images = self.up(images)
        images = torch.cat([images, skip], dim=1)  # Concatenate skip connection
        images = self.conv(images)

        return images


class UNet(nn.Module):
    def __init__(
        self,
        contrasts=1,
        features=[64, 128, 256, 512],
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Initial Conv
        self.init_conv = nn.Conv2d(
            contrasts,
            features[0],
            kernel_size=1,
            device=device,
            dtype=dtype,
        )

        # Encoder (Down Path)
        for i in range(len(features)):
            self.encoder.append(
                DoubleConv(
                    features[i],
                    features[i],
                    device=device,
                    dtype=dtype,
                ),
            )

        # Down Conv blocks for downsampling
        self.down_convs = nn.ModuleList()
        for i in range(len(features)-1):
            self.down_convs.append(
                DownConvBlock(
                    features[i],
                    features[i+1],
                    device=device,
                    dtype=dtype,
                ),
            )

        # Bottleneck
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1],
            device=device,
            dtype=dtype,
        )

        # Decoder (Up Path)
        for i in range(len(features)-1, 0, -1):
            self.decoder.append(
                UpConvBlock(
                    features[i],
                    features[i-1],
                    device=device,
                    dtype=dtype,
                ),
            )

        # Final Conv
        self.final_conv = nn.Conv2d(
            features[0],
            contrasts,
            kernel_size=1,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        images,
    ):

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

        # ---

        skips = []

        # Initial Conv
        images = self.init_conv(images)

        # Encoder
        for i, layer in enumerate(self.encoder):
            images = layer(images)
            if i < len(self.encoder) - 1:  # Skip the last layer (bottleneck input)
                skips.append(images)
                images = self.down_convs[i](images)  # Down Conv for downsampling

        # Bottleneck
        images = self.bottleneck(images)

        # Decoder
        skips = skips[::-1]  # Reverse for upsampling
        for i, layer in enumerate(self.decoder):
            images = layer(images, skips[i])

        # Final Conv
        images = self.final_conv(images)

        # ---

        images = torch.unflatten(
            images, 1, (nTI, nTE)
        )  # switches shape to [blank, nTI, nTE, nX, nY]
        images = torch.permute(
            images, (0, 3, 4, 1, 2)
        )  # switches shape to [blank, nX, nY, nTI, nTE]
        images = images[
            0, :, :, None, :, :
        ]  # switches shape back to [nX, nY, nZ, nTI, nTE]

        return images
