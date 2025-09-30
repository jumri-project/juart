import torch
import torch.nn as nn
import sys
sys.path.insert(0, "../../src")
from juart.conopt.functional.__init__ import pad_tensor, crop_tensor
from common import ComplexActivation, ConvLayer


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
        contrasts: int = 1,
        features: list[int] = [64, 128, 256, 512],
        kernel_size: tuple[int] = (3,3),
        device: str = None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.kernel = kernel
        self.device = device
        self.dtype = dtype
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Initial Conv
        self.init_conv = nn.Conv2d(
            contrasts,
            features[0],
            kernel_size=kernel_size,
            device=device,
            dtype=dtype,
        )

        # Encoder (Down Path)
        for i in range(len(features)):
            self.encoder.append(
                DoubleConv(
                    features[i],
                    features[i],
                    kernel_size = kernel_size,
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
            kernel_size = kernel_size,
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
            kernel_size = kernel_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        images,
    ):

        if len(self.kernel) == 2:
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


        elif len(self.kernel) == 3:
            
            nTI, nTE = images.shape[-2:]
    
            images = torch.permute(
                images, (3, 4, 0, 1, 2)
            )  # switches shape to [nTI, nTE, nX, nY, nZ]
            images = torch.flatten(
                images, start_dim=0, end_dim=1
            )  # switches shape to [nTI*nTE, nX, nY, nZ]
    
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
            )  # switches shape to [nTI, nTE, nX, nY, nZ]
            images = torch.permute(
                images, (2, 3, 4, 0, 1)
            )  # switches shape to [nX, nY, nZ, nTI, nTE]
        
        return images
