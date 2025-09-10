from typing import Tuple

import torch
import torch.nn as nn
from torch import jit
from tqdm import tqdm

from ..utils.fourier import apply_transfer_function, nonuniform_transfer_function
from ..utils.validation import timing_layer, validation_layer


@jit.script
def inner_product(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return torch.vdot(x.flatten(), y.flatten()).real


def conj_grad(
    A: nn.Module, b: torch.Tensor, x: torch.Tensor, niter: int, verbose: bool = True
) -> torch.Tensor:
    r = b - A(x)
    p = r

    rsnot = inner_product(r, r)
    rsold, rsnew = rsnot, rsnot

    log = tqdm(
        total=niter,
        desc="CG",
        disable=(not verbose),
    )

    for iter in range(niter):
        Ap = A(p)
        pAp = inner_product(p, Ap)
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = inner_product(r, r)
        beta = rsnew / rsold
        rsold = rsnew
        p = r + beta * p

        str_out = "[CG] "
        str_out += f"Iter: {iter:0>{len(str(niter))}} "
        str_out += f"Res: {rsnew:.2E} "
        log.set_description_str(str_out)
        log.update(1)

    return x


class ToeplitzOperator(nn.Module):
    def __init__(
        self,
        shape,
        axes: Tuple[int] = (1, 2),
        device=None,
        dtype=torch.complex64,
    ):
        super().__init__()

        self.axes = axes
        self.shape = shape
        self.device = device
        self.dtype = dtype

        self.num_chunks = 1

    def init(
        self,
        lamda: torch.Tensor,
        kspace_trajectory: torch.Tensor,
        kspace_mask: torch.Tensor = None,
        sensitivity_maps: torch.Tensor = None,
    ):
        self.kernel = nonuniform_transfer_function(
            kspace_trajectory,
            (1,) + self.shape,
            weights=kspace_mask,
        )
        self.kernel = self.kernel / 4

        self.sensitivity_maps = sensitivity_maps
        self.lamda = lamda

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        # Multiply with sensitivity maps
        futures = []

        sensitivity_maps = torch.chunk(
            self.sensitivity_maps,
            self.num_chunks,
            dim=1,
        )

        for sensitivity_maps_set in sensitivity_maps:
            v = images

            v = v * sensitivity_maps_set[..., None, None]

            v = apply_transfer_function(
                v,
                self.kernel,
                axes=self.axes,
            )

            v = v * torch.conj(sensitivity_maps_set[..., None, None])

            futures.append(v)

        v = torch.sum(torch.cat(futures, dim=0), dim=0)

        return v + self.lamda * images


class DataConsistency(nn.Module):
    def __init__(
        self,
        shape,
        axes: Tuple[int] = (1, 2),
        niter=10,
        lamda_start=0.05,
        timing_level=0,
        validation_level=0,
        device=None,
        dtype=torch.complex64,
        verbose: bool = False,
    ):
        super().__init__()

        self.toep_ob = ToeplitzOperator(
            shape,
            axes=axes,
            device=device,
            dtype=dtype,
        )
        self.lam = nn.Parameter(
            torch.tensor(
                lamda_start,
                dtype=torch.float32,
                device=device,
            )
        )
        self.niter = niter
        self.timing_level = timing_level
        self.validation_level = validation_level
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

    @timing_layer
    @validation_layer
    def init(
        self,
        images_regridded: torch.Tensor,
        kspace_trajectory: torch.Tensor,
        kspace_mask: torch.Tensor = None,
        sensitivity_maps: torch.Tensor = None,
    ):
        images_regridded = images_regridded.to(self.device)
        kspace_trajectory = kspace_trajectory.to(self.device)
        if kspace_mask is not None:
            kspace_mask = kspace_mask.to(self.device)
        if sensitivity_maps is not None:
            sensitivity_maps = sensitivity_maps.to(self.device)

        self.images_regridded = images_regridded.clone().detach()
        self.toep_ob.init(
            self.lam.data,
            kspace_trajectory,
            kspace_mask=kspace_mask,
            sensitivity_maps=sensitivity_maps,
        )

    @timing_layer
    @validation_layer
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)

        images = conj_grad(
            self.toep_ob,
            self.images_regridded + self.lam * images,
            images,
            self.niter,
            verbose=self.verbose,
        )

        return images
