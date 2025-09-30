import torch
from torch import nn


class ComplexActivation(nn.Module):
    def __init__(
        self,
        activation
    ):
        super().__init__()
        match activation:

            case "ReLU":
                self.functional = nn.ReLU()

            case "ELU":
                self.functional = nn.ELU()

            case "Identity":
                self.functional = nn.Identity()

            case _:
                raise ValueError(f'activation function {activation} is unknown.')

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return torch.complex(
            self.functional(images.real),
            self.functional(images.imag),
        )



class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: tuple[int] = (
            3,
            3,
        ),
        padding=1,
        bias=False,
        activation="ReLU",
        device=None,
        dtype=torch.complex64,
    ):
        """
        Initializes a Convolutional Layer. Dependent on the property kernel length
        it will initialize a 2D or 3D Convolutional Layer. The other properties
        are the same for 2D and 3D.

        Parameters
        ----------
        in_channles : int
            number of channels in the input image (contrasts).
        out_channels: int
            number of channels produced by the convolution.
        kernel_size : Tuple[int], optional
            forms the tuple that is used for the convolutions and defines
            whether a 2D or 3D ConvLayer should be initialized (X, Y, Z)
            (default is (3,3)).
        padding: int, optional
            The Padding thats added to all four sides of the input (default is 1).
        bias: bool, optional
            If True it will add a learnable bias to the output (default is False)
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
        self.dtype = dtype

        if len(kernel_size) == 2:
            self.convlayer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=bias,
                device=device,
                dtype=dtype,
            )

        elif len(kernel_size) == 3:
            self.convlayer = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=bias,
                device=device,
                dtype=dtype,
            )

        self.activation = ComplexActivation(activation)

    def forward(self, image):
        image = self.convlayer(image)
        image = self.activation(image)

        return image


class DoubleConv(nn.Module):
    def __init__(
        self,
        features: int,
        kernel_size: tuple[int] = (3, 3),
        activation: list[str] = 'ReLU',
        padding: int = 1,
        bias: bool = False,
        device: str = None,
        dtype=torch.complex64,
    ):
        """
        Initializes a DoubleConvolution Class and defines all its parameters used for convolution.
        Every ResNetBlock contains out of 2 convolutional Layers.

        Parameters
        ----------
        features : int
            number of the in- and output features for the ConvLayer. The more features it has the more
            things it can learn.
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

        if type(activation) is str:
            activation = [activation, activation]

        super().__init__()


        self.double_conv = nn.Sequential(
            ConvLayer(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                activation=activation[0],
                device=device,
                dtype=dtype
            ),
            ConvLayer(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                activation=activation[1],
                device=device,
                dtype=dtype
            )
        )

    def forward(
        self,
        images,
    ):
        return self.double_conv(images)