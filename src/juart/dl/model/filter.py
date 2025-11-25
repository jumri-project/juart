from typing import Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...conopt.functional.fourier import (
    fourier_transform_adjoint,
    fourier_transform_forward,
)


class FilterClass(nn.Module):

    def __init__(
        self,
        filter_name: str = None,
        radius: int = 0,
        axis: tuple[int] = (1, 2, 3),
        device="cpu"
    ):
        super().__init__()
        FilterList = ["None", "LowPass"]

        if filter_name is None or filter_name == "None":
            self.filter = NoFilter()

        elif filter_name == "LowPass":
            self.filter = LowPassFilter(
                radius=radius,
                axis=axis,
                device=device
            )

        else:
            raise ValueError(f"unexpected filter name. Try {FilterList}")

    def forward(self, image):
        image = self.filter(image)
        return image


class LowPassFilter(nn.Module):

    def __init__(
        self,
        radius: int,
        axis: tuple[int] = (1, 2, 3),
        device="cpu"
    ):
        super().__init__()
        self.radius = radius
        self.axis = axis
        self.device = device

        if radius <= 1:
            raise ValueError("Invalid Value for filter_radius")

    def forward(self, image):

        image = image[None, ..., 0, 0]

        print(image.shape)
        FT = fourier_transform_forward(image, self.axis).to(self.device)

        if len(self.axis) == 3:
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(FT.shape[0]),
                    torch.arange(FT.shape[0]),
                    torch.arange(FT.shape[0]),
                    indexing='ij'),
                dim=-1).to(self.device)

        elif len(self.axis) == 2:
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(FT.shape[0]),
                    torch.arange(FT.shape[0]),
                    indexing='ij'),
                dim=-1).to(self.device)

        center = (FT.shape[0]) / 2.0
        distance = torch.sqrt(torch.sum((coords - center)**2, dim=-1)).to(self.device)

        mask = (distance <= self.radius).float().to(self.device)
        low_pass_image = fourier_transform_adjoint(FT * mask, self.axis)[0,:,:,:,None,None].to(self.device)
        print(low_pass_image.shape)

        return low_pass_image


class NoFilter(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image):
        return image