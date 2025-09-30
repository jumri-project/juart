import torch
from torch import nn

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



class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Tuple[int] = (
            3,
            3,
        ),
        padding=1,
        bias=False,
        activation="ReLU",
        device=None,
        ConvLayerCheckpoints = False,
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
        self.ConvLayerCheckpoints = ConvLayerCheckpoints
        self.dtype = dtype
        
        if len(kernel_size) == 2:
            self.convlayer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                activation=activation,
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
                activation=activation,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            

    def forward(
        self,
        image):

        if self.ConvLayerCheckpoints:
            image = checkpoint(self.CalcLayer, image, use_reentrant = False)

        else:
            image = image.to(dtype=self.dtype)
            image = self.convlayer(image)

        return image

    def CalcLayer(self,image):

        image = self.convlayer(image)

        return image



class DoubleConv(nn.Module):
    def __init__(
        self,
        features: int,
        kernel_size= tuple[int] = (3, 3),
        activation: list[str] = 'ReLU',
        padding: int = 1,
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

        if type(activation) == str:
            activation = [activation, activation]

        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                features,
                features,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation[0],
                device=device,
                dtype=dtype,
            ),
            nn.Conv2d(
                features,
                features,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation[1],
                device=device,
                dtype=dtype,
            )
        )

    def forward(
        self,
        images,
    ):
        return self.double_conv(images)