import time

import numpy as np
import torch
import zarr
from s3fs import S3FileSystem

from ..utils.fourier import nonuniform_fourier_transform_adjoint


class DatasetInference(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        datasets,
        slices,
        num_spokes,
        root_dir="",
        endpoint_url="",
        backend=None,
        device=None,
    ):
        self.path = path
        self.datasets = datasets
        self.slices = slices
        self.num_spokes = num_spokes

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

        toc = time.time() - tic
        print(f"Data - Completed loading dataset in {toc:.1f} seconds.")

        tic = time.time()
        print("Data - Started regridding dataset ...")

        # TODO: Think about batch processing
        # TODO: finufft is 50% slower than torchkbnufft
        # TODO: Scaling with 0.5 is necessary to stay compatible with torchkbnufft

        images_regridded = nonuniform_fourier_transform_adjoint(
            kspace_trajectory,
            kspace_data,
            n_modes=(nX, nY),
        )

        images_regridded = 0.5 * torch.sum(
            images_regridded * torch.conj(sensitivity_maps[..., None, None]),
            dim=0,
        )

        toc = time.time() - tic
        print(
            f"Data - Completed regridding dataset {images_regridded.shape} in {toc:.1f} seconds."
        )

        return {
            "images_regridded": images_regridded,
            "kspace_trajectory": kspace_trajectory,
            "sensitivity_maps": sensitivity_maps,
        }

    def __getitem__(self, index):
        if isinstance(index, int):
            output = self.get_index(index)

        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if step is None:
                step = 1

            output = {
                "images_regridded": list(),
                "kspace_trajectory": list(),
                "sensitivity_maps": list(),
            }

            for ii in range(start, stop, step):
                output_ii = self.get_index(ii)

                for key in output.keys():
                    output[key].append(output_ii[key])

            for key in output.keys():
                output[key] = torch.cat(output[key], dim=0)

        else:
            raise IndexError("Invalid index type. Use either an integer or a slice.")

        return output


class ImageStore:
    def __init__(
        self,
        path,
        datasets,
        slices,
        tag,
        array_name="images",
        array_shape=(256, 256, 1, 160, 19, 9),
        array_chunks=(256, 256, 1, 1, 1, 1),
        array_dtype=np.complex64,
        endpoint_url="",
        backend="local",
    ):
        if backend == "local":
            self.store = zarr.storage.LocalStore(
                # path=path,
            )

        elif backend == "s3":
            # Environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set.
            fs = S3FileSystem(
                anon=False,
                endpoint_url=endpoint_url,
                asynchronous=True,
            )
            self.store = zarr.storage.FsspecStore(
                fs,
                # path=path,
            )

        self.path = path
        self.datasets = datasets
        self.slices = slices
        self.tag = tag

        self.array_name = array_name
        self.array_shape = array_shape
        self.array_chunks = array_chunks
        self.array_dtype = array_dtype

    def create(self, dataset_index, overwrite=False):
        path = self.path % (self.datasets[dataset_index], self.tag)

        print(f"Store - Creating array: {path}/{self.array_name}")

        self.store.path = path

        self.root = zarr.open_group(
            store=self.store,
            mode="a",
        )

        self.root.create_array(
            name=self.array_name,
            shape=self.array_shape,
            chunks=self.array_chunks,
            dtype=self.array_dtype,
            overwrite=overwrite,
        )

        self.store.path = ""

    def save(self, images, index):
        assert isinstance(index, int)

        dataset_index = index // len(self.slices)
        slice_index = np.mod(index, len(self.slices))

        print(
            f"Store - Saving Images {self.datasets[dataset_index]} - Slice {self.slices[slice_index]} ..."
        )

        self.group = zarr.open_group(
            store=self.store,
            path=self.path % (self.datasets[dataset_index], self.tag),
            mode="a",
        )

        images = images.cpu().numpy()

        print("Image Store", images.shape)
        # images.shape (256, 256, 1, 19, 9)

        images = images[:, :, :, None, :, :]
        # images = np.transpose(images, (1, 2, 3, 0, 4, 5))

        self.group[self.array_name][
            :, :, :, self.slices[slice_index] : self.slices[slice_index] + 1, :, :
        ] = images
