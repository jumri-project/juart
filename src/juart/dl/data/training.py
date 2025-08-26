import time

import numpy as np
import torch
import torch.distributed as dist
import zarr
from s3fs import S3FileSystem

from ..utils.fourier import nonuniform_fourier_transform_adjoint
from ..utils.sampling import uniform_selections


class DatasetTraining(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        datasets,
        slices,
        num_spokes,
        split_fractions,
        root_dir="",
        endpoint_url="",
        group_rank=0,
        mode="",
        backend=None,
        device=None,
    ):
        self.path = path
        self.datasets = datasets
        self.slices = slices
        self.num_spokes = num_spokes
        self.split_fractions = split_fractions
        self.group_rank = group_rank
        self.mode = mode
        self.device = device

        self.keys = [
            "images_regridded",
            "kspace_trajectory",
            "kspace_data",
            "kspace_mask_source",
            "kspace_mask_target",
            "sensitivity_maps",
        ]

        if backend == "local":
            self.store = zarr.storage.LocalStore(root_dir)

        elif backend == "s3":
            # Environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set.
            fs = S3FileSystem(
                anon=False,
                endpoint_url=endpoint_url,
                asynchronous=True,
            )
            self.store = zarr.storage.FsspecStore(
                fs,
                read_only=True,
            )

        self.device = device

    def __len__(self):
        return len(self.datasets) * len(self.slices)

    def get_index(self, index):
        assert isinstance(index, int)

        dataset_index = index // len(self.slices)
        slice_index = np.mod(index, len(self.slices))

        tic = time.time()
        print(
            f"Data - Started loading Dataset {self.datasets[dataset_index]} - Slice {self.slices[slice_index]} ..."
        )

        zarr_preproc_file = zarr.open_group(
            self.store,
            path=self.path % self.datasets[dataset_index],
            mode="r",
        )

        nC, nX, nY, nZ, nS = zarr_preproc_file["C"].shape[:5]
        nC, spokes, baseresolution, nZ, nS, nTI, nTE = zarr_preproc_file["d"].shape

        # Read data
        sensitivity_maps = zarr_preproc_file["C"][
            :, :, :, :, self.slices[slice_index], 0, 0
        ]
        kspace_trajectory = zarr_preproc_file["k"][:, : self.num_spokes, :, 0, 0, :, :]
        kspace_data = (
            zarr_preproc_file["d"][
                :, : self.num_spokes, :, 0, self.slices[slice_index], :, :
            ]
            / 1e-4
        )

        kspace_data = torch.tensor(
            kspace_data, dtype=torch.complex64, device=self.device
        )
        kspace_trajectory = torch.tensor(
            kspace_trajectory, dtype=torch.float32, device=self.device
        )
        sensitivity_maps = torch.tensor(
            sensitivity_maps, dtype=torch.complex64, device=self.device
        )

        kspace_data = torch.flatten(kspace_data, start_dim=1, end_dim=2)
        kspace_trajectory = torch.flatten(kspace_trajectory, start_dim=1, end_dim=2)

        nK = kspace_trajectory.shape[1]

        toc = time.time() - tic
        print(f"Data - Completed loading dataset in {toc:.1f} seconds.")

        tic = time.time()
        print(
            f"Rank {dist.get_rank()} - Data - Started creating masks {self.split_fractions} ..."
        )

        kspace_mask = torch.zeros(
            (len(self.split_fractions), 1, nK, nTI, nTE),
            dtype=torch.float32,
            device=self.device,
        )

        for iTI in range(nTI):
            for iTE in range(nTE):
                # Create unique seed for every combination of dataset, slice, inversion and echo
                seed = (
                    (dataset_index * len(self.slices) + slice_index) * nTI + iTI
                ) * nTE + iTI

                indices = uniform_selections(nK, self.split_fractions, seed=seed)[:-1]

                for batch, index in enumerate(indices):
                    kspace_mask[batch, :, index, iTI, iTE] = 1

        # Training process
        # ----------------
        # We create N sets of masks, each consisting of a defined fraction of
        # k-space data. The first set is reserved for validation, the other
        # sets are used for training.
        # --- During training ---
        # Each training process (i.e. group rank) takes a set
        # (kspace_mask_input) and regrids the data from this set. It then
        # reconstructs the missing data using the reconstruction network.
        # In the loss function, it predicts the data from all other training.
        # sets combined (kspace_mask_training).
        # --- During validation ---
        # Each training process (i.e. group rank), takes a set
        # (kspace_mask_input) and regrids the data from this set. It then
        # reconstructs the missing data using the reconstruction network.
        # In the loss function, it predicts the data from the validation set.

        kspace_mask_validation = kspace_mask[:1, :, :, :, :]
        kspace_mask_input = kspace_mask[1:, :, :, :, :]
        kspace_mask_training = 1 - kspace_mask_input - kspace_mask_validation

        if self.mode == "training":
            kspace_mask_source = kspace_mask_input[self.group_rank, :, :, :, :]
            kspace_mask_target = kspace_mask_training[self.group_rank, :, :, :, :]

        elif self.mode == "validation":
            kspace_mask_source = kspace_mask_input[self.group_rank, :, :, :, :]
            kspace_mask_target = kspace_mask_validation

        source_fractions = kspace_mask_source.sum() / (nK * nTI * nTE)
        target_fractions = kspace_mask_target.sum() / (nK * nTI * nTE)

        print(f"Rank {dist.get_rank()} - Data - Source fractions {source_fractions}.")
        print(f"Rank {dist.get_rank()} - Data - Target fractions {target_fractions}.")

        toc = time.time() - tic
        print(
            f"Rank {dist.get_rank()} - Data - Completed creating mask {kspace_mask.shape} in {toc:.1f} seconds."
        )

        tic = time.time()
        print(f"Rank {dist.get_rank()} - Data - Started regridding dataset ...")

        # TODO: Think about batch processing
        # TODO: finufft is 50% slower than torchkbnufft
        # TODO: Scaling with 0.5 is necessary to stay compatible with torchkbnufft

        images_regridded = nonuniform_fourier_transform_adjoint(
            kspace_trajectory,
            kspace_data * kspace_mask_source,
            n_modes=(nX, nY),
        )

        images_regridded = 0.5 * torch.sum(
            images_regridded * torch.conj(sensitivity_maps[..., None, None]),
            dim=0,
        )

        toc = time.time() - tic
        print(
            f"Rank {dist.get_rank()} - Data - Completed regridding dataset {images_regridded.shape} in {toc:.1f} seconds."
        )

        return {
            "images_regridded": images_regridded,
            "kspace_trajectory": kspace_trajectory,
            "kspace_data": kspace_data,
            "kspace_mask_source": kspace_mask_source,
            "kspace_mask_target": kspace_mask_target,
            "sensitivity_maps": sensitivity_maps,
        }

    def get_indices(self, indices):
        output = dict()

        for key in self.keys:
            output[key] = list()

        for ii in range(start, stop, step):
            output_ii = self.get_index(ii)

            for key in self.keys:
                output[key].append(output_ii[key])

        for key in self.keys:
            output[key] = torch.cat(output[key], dim=0)

    def __getitem__(self, index):
        if isinstance(index, int):
            output = self.get_index(index)

        elif isinstance(index, tuple) or isinstance(index, list):
            output = dict()

            for key in self.keys:
                output[key] = list()

            for ii in index:
                output_ii = self.get_index(ii)

                for key in self.keys:
                    output[key].append(output_ii[key])

            for key in self.keys:
                output[key] = torch.cat(output[key], dim=0)

        else:
            raise IndexError("Invalid index type. Use either an integer or a tuple.")

        return output
