import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from ..utils.validation import timing_layer, validation_layer


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
        return torch.complex(self.functional(images.real), self.functional(images.imag))


class StableConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.epsilon = epsilon

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        weight_mean = self.weight.mean(
            dim=(1, 2, 3),
            keepdim=True,
        )

        weight_std = (
            self.weight.std(
                dim=(1, 2, 3),
                keepdim=True,
            )
            + self.epsilon
        )

        weight_standardized = (self.weight - weight_mean) / weight_std

        return nn.functional.conv2d(
            images,
            weight_standardized,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        bias=False,
        weight_standardization=False,
        spectral_normalization=False,
        activation="ReLU",
        device=None,
        dtype=torch.complex64,
    ):
        # conv2d = StableConv2D(
        #     in_channels,
        #     out_channels,
        #     kernel_size,
        #     padding=padding,
        #     bias=bias,
        #     weight_standardization=weight_standardization,
        #     device=device,
        #     dtype=dtype,
        # )
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if spectral_normalization:
            conv2d = spectral_norm(conv2d)

        if activation == "Identity":
            function = nn.Identity()
        else:
            function = ComplexActivation(activation)

        super().__init__(conv2d, function)


class ResNetBlock(nn.Sequential):
    def __init__(
        self,
        features,
        kernel_size,
        weight_standardization=False,
        spectral_normalization=False,
        activation="ReLU",
        device=None,
        dtype=torch.complex64,
    ):
        conv_layer_1 = ConvLayer(
            features,
            features,
            kernel_size,
            weight_standardization=weight_standardization,
            spectral_normalization=spectral_normalization,
            activation=activation,
            device=device,
            dtype=dtype,
        )

        conv_layer_2 = ConvLayer(
            features,
            features,
            kernel_size,
            weight_standardization=weight_standardization,
            spectral_normalization=spectral_normalization,
            activation="Identity",
            device=device,
            dtype=dtype,
        )

        super().__init__(conv_layer_1, conv_layer_2)


class ResNetBlocksModule(nn.Module):
    def __init__(
        self,
        features,
        kernel_size=3,
        weight_standardization=False,
        spectral_normalization=False,
        activation="ReLU",
        num_of_resblocks=15,
        scale_factor=0.1,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ResNetBlock(
                    features,
                    kernel_size,
                    weight_standardization=weight_standardization,
                    spectral_normalization=spectral_normalization,
                    activation=activation,
                    device=device,
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
        for layer in self.layers:
            images = images + self.scale_factor * layer(images)

        return images


class ResNet(nn.Module):
    def __init__(
        self,
        contrasts=1,
        features=128,
        num_of_resblocks=15,
        kernel_size=3,
        weight_standardization=False,
        spectral_normalization=False,
        activation="ReLU",
        timing_level=0,
        validation_level=0,
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.layer1 = ConvLayer(
            contrasts,
            features,
            kernel_size,
            weight_standardization=weight_standardization,
            spectral_normalization=spectral_normalization,
            activation="Identity",
            device=device,
            dtype=dtype,
        )
        self.layer2 = ResNetBlocksModule(
            features,
            kernel_size,
            weight_standardization=weight_standardization,
            spectral_normalization=spectral_normalization,
            activation=activation,
            num_of_resblocks=num_of_resblocks,
            device=device,
            dtype=dtype,
        )
        self.layer3 = ConvLayer(
            features,
            features,
            kernel_size,
            weight_standardization=weight_standardization,
            spectral_normalization=spectral_normalization,
            activation="Identity",
            device=device,
            dtype=dtype,
        )
        self.layer4 = ConvLayer(
            features,
            contrasts,
            kernel_size,
            weight_standardization=weight_standardization,
            spectral_normalization=spectral_normalization,
            activation="Identity",
            device=device,
            dtype=dtype,
        )

        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device

    @timing_layer
    @validation_layer
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)

        nTI, nTE = images.shape[-2:]

        images = images[None, :, :, 0, :, :]
        images = torch.permute(images, (0, 3, 4, 1, 2))
        images = torch.flatten(images, start_dim=1, end_dim=2)

        l1_out = self.layer1(images)
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        images = self.layer4(l3_out + l1_out)

        images = torch.unflatten(images, 1, (nTI, nTE))
        images = torch.permute(images, (0, 3, 4, 1, 2))
        images = images[0, :, :, None, :, :]

        return images
