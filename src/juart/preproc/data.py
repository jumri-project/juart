import ctypes
import multiprocessing as mp
from functools import partial

import ismrmrd
import numpy as np
import torch
from tqdm.auto import tqdm

from ..conopt.functional.fourier import (
    fourier_transform_adjoint,
)
from ..parim.gcc import (
    apply_geometric_coil_compression,
    geometric_coil_compression_matrices,
)
from ..recon.grappa import grappa

# from .aux import espirit_sense, sake_espirit
from .aux import sake_espirit, sake_espirit_multicontrast

# Global variables that the worker processes will access.
# These are set via the worker_init function.
shared_dataset = None
shared_kdata_buffer = None
shared_sensmaps_buffer = None


def get_shape(dataset):
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())

    NCha, NCol = dataset.read_acquisition(0).data.shape
    NLin = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
    NPar = header.encoding[0].encodingLimits.kspace_encoding_step_2.maximum + 1
    NSli = header.encoding[0].encodingLimits.slice.maximum + 1
    NSet = header.encoding[0].encodingLimits.set.maximum + 1
    NEco = header.encoding[0].encodingLimits.contrast.maximum + 1

    return NCha, NCol, NLin, NPar, NSli, NSet, NEco


class KSpaceData:
    def __init__(
        self,
        kdata_shape,
        kdata_dtype=torch.complex64,
        buffer_dtype=ctypes.c_double,
    ):
        """
        Initialize the KSpaceData.

        Parameters:
            kdata_raw_shape: Tuple specifying the shape
            (e.g., (NCha, NCol, NLin, NPar, NSli, NSet, NEco)).
            kdata_raw_dtype: The torch data type (e.g., torch.complex64).
        """

        self.kdata_shape = kdata_shape
        self.kdata_dtype = kdata_dtype
        self.buffer_dtype = buffer_dtype

        self.kdata_buffer = mp.RawArray(
            self.buffer_dtype,
            int(np.prod(kdata_shape)),
        )

        self.kdata = torch.frombuffer(
            self.kdata_buffer,
            dtype=kdata_dtype,
        ).reshape(kdata_shape)

    def index_data(self, index):
        kdata = self.kdata[index]

        new_kdata_shape = kdata.shape
        new_kdata_buffer = mp.RawArray(
            self.buffer_dtype,
            int(np.prod(new_kdata_shape)),
        )

        new_kdata = torch.frombuffer(
            new_kdata_buffer,
            dtype=self.kdata_dtype,
        ).reshape(new_kdata_shape)

        new_kdata.copy_(kdata)

        self.kdata = new_kdata
        self.kdata_buffer = new_kdata_buffer
        self.kdata_shape = new_kdata_shape

    def swapaxes(self, axis0, axis1):
        kdata = self.kdata.swapaxes(axis0, axis1)

        self.kdata_shape = kdata.shape
        self.kdata = self.kdata.reshape(kdata.shape)
        self.kdata[:] = kdata.clone()

    @staticmethod
    def worker_init(
        dataset,
        kdata_buffer,
        sensmaps_buffer,
    ):
        """
        This initializer runs in each worker process. It sets the global variables
        so that process_acquisition can access the shared objects.
        """
        global shared_dataset, shared_kdata_buffer, shared_sensmaps_buffer
        shared_dataset = dataset
        shared_kdata_buffer = kdata_buffer
        shared_sensmaps_buffer = sensmaps_buffer

    @staticmethod
    def run_task(initargs, func, num_indices):
        # Initialize the pool. The initializer sets the global variables in each child
        # process.
        with mp.Pool(
            processes=mp.cpu_count() // 2,
            initializer=KSpaceData.worker_init,
            initargs=initargs,
        ) as pool:
            with tqdm(total=num_indices) as progress_bar:
                for _ in pool.imap_unordered(
                    func,
                    range(num_indices),
                ):
                    progress_bar.update()

    @staticmethod
    def process_acquisition(
        index,
        shared_kdata_shape,
        shared_kdata_dtype,
    ):
        """
        This function is executed in each worker process. It reconnects to the shared
        memory using the global variables, reads an acquisition, and updates the shared
        tensor.
        """

        # Reconnect to shared memory using the global variables.
        kdata = torch.frombuffer(
            shared_kdata_buffer,
            dtype=shared_kdata_dtype,
        ).reshape(shared_kdata_shape)

        # Read the acquisition.
        acquisition = shared_dataset.read_acquisition(index)

        if acquisition.data.shape == kdata.shape[:2]:
            # Update shared memory.
            kdata[
                :,
                :,
                acquisition.idx.kspace_encode_step_1,
                acquisition.idx.kspace_encode_step_2,
                acquisition.idx.slice,
                acquisition.idx.set,
                acquisition.idx.contrast,
            ] = torch.tensor(acquisition.data)

    def read_data(self, dataset, is_pulseq=False):
        """
        Launch the multiprocessing pool to process all acquisitions.
        """
        num_indices = dataset.number_of_acquisitions()

        func = partial(
            KSpaceData.process_acquisition,
            shared_kdata_shape=self.kdata_shape,
            shared_kdata_dtype=self.kdata_dtype,
        )

        KSpaceData.run_task(
            (dataset, self.kdata_buffer, None),
            func,
            num_indices,
        )

        # Pulseq specific: flip readout and partition axes
        if is_pulseq:
            self.kdata[:] = torch.flip(self.kdata, (1, 3))

    @staticmethod
    def process_grappa(
        index,
        shared_kdata_shape,
        shared_kdata_dtype,
    ):
        NAcl = 32

        NCha, NCol, NLin, NPar, NSli, NSet, NEco = shared_kdata_shape

        ILin, ISet, IEco = torch.unravel_index(torch.tensor(index), (NLin, NSet, NEco))

        # Reconnect to shared memory
        kdata = torch.frombuffer(
            shared_kdata_buffer,
            dtype=shared_kdata_dtype,
        ).reshape(shared_kdata_shape)

        # WIP
        #
        # NUsf = 2
        #
        # partition_mask = torch.zeros((NPar,), dtype=torch.float32)
        # partition_mask[::NUsf] = 1
        # partition_mask[(NPar - NAcl) // 2 : (NPar + NAcl) // 2] = 1
        # partition_mask = partition_mask[None, None, None, :, None, None, None]
        #
        # kdata_tmp = kdata[:, :, ILin : ILin + 1, :, :,
        # .                  ISet : ISet + 1, IEco : IEco + 1]
        #
        # # Now swap line and partition dimension
        # partition_mask = partition_mask.swapaxes(2, 3)
        # kdata_tmp = kdata_tmp.swapaxes(2, 3)
        # kdata_tmp = espirit_sense(kdata_tmp, partition_mask)
        # # Swap back
        # kdata_tmp = kdata_tmp.swapaxes(2, 3)
        #
        # kdata[:, :, ILin : ILin + 1, :, :,
        # .      ISet : ISet + 1, IEco : IEco + 1] = kdata_tmp

        kdata[:, :, ILin, :, 0, ISet, IEco] = grappa(
            kdata[:, :, ILin, :, 0, ISet, IEco], NAcl, kernel_size=(5, 5), coil_axis=0
        )

    def apply_mask(self, NUsf, NAcl, is_pulseq=False):
        NCha, NCol, NLin, NPar, NSli, NSet, NEco = self.kdata_shape

        partition_mask = torch.zeros((NPar,), dtype=torch.float32)
        partition_mask[::NUsf] = 1
        partition_mask[(NPar - NAcl) // 2 : (NPar + NAcl) // 2] = 1

        if is_pulseq:
            partition_mask = torch.flip(partition_mask, (0,))

        # Retrospectively accelerate along partition dimension
        self.kdata *= partition_mask[None, None, None, :, None, None, None]

    def reconstruct_partitions(self):
        NCha, NCol, NLin, NPar, NSli, NSet, NEco = self.kdata_shape

        num_indices = NLin * NSet * NEco
        func = partial(
            KSpaceData.process_grappa,
            shared_kdata_shape=self.kdata_shape,
            shared_kdata_dtype=self.kdata_dtype,
        )

        KSpaceData.run_task(
            (None, self.kdata_buffer, None),
            func,
            num_indices,
        )

    def compression_matrix(self, NCha_comp, ISet_comp, IEco_comp):
        kdata = self.kdata[:, :, :, :, :, ISet_comp, IEco_comp]
        kdata = fourier_transform_adjoint(kdata, axes=4)

        return geometric_coil_compression_matrices(kdata, NCha_comp)

    @staticmethod
    def process_fourier_coil_compression(
        index,
        shared_comp_matrix,
        shared_kdata_shape,
        shared_kdata_dtype,
    ):
        NCha, NCol, NLin, NPar, NSli, NSet, NEco = shared_kdata_shape
        NCha_comp, NCha_mat, NSli_mat = shared_comp_matrix.shape

        ILin, ISet, IEco = torch.unravel_index(torch.tensor(index), (NLin, NSet, NEco))

        # Reconnect to shared memory
        kdata = torch.frombuffer(
            shared_kdata_buffer,
            dtype=shared_kdata_dtype,
        ).reshape(shared_kdata_shape)

        kdata_tmp = kdata[:, :, ILin : ILin + 1, :, :, ISet : ISet + 1, IEco : IEco + 1]
        kdata_tmp = fourier_transform_adjoint(kdata_tmp, axes=4)
        kdata_tmp = apply_geometric_coil_compression(
            kdata_tmp, shared_comp_matrix, NCha_comp
        )
        kdata[
            :NCha_comp, :, ILin : ILin + 1, :, :, ISet : ISet + 1, IEco : IEco + 1
        ] = kdata_tmp

    def compress_data(self, comp_matrix):
        NCha, NCol, NLin, NPar, NSli, NSet, NEco = self.kdata_shape
        NCha_comp, NCha_mat, NSli_mat = comp_matrix.shape
        # NCha and NCha_mat must be identical
        # NSli and NSli_mat must be identical
        # Otherwise comp_matrix does not fit to data

        num_indices = NLin * NSet * NEco
        func = partial(
            KSpaceData.process_fourier_coil_compression,
            shared_comp_matrix=comp_matrix,
            shared_kdata_shape=self.kdata_shape,
            shared_kdata_dtype=self.kdata_dtype,
        )

        KSpaceData.run_task(
            (None, self.kdata_buffer, None),
            func,
            num_indices,
        )

        self.index_data([slice(0, NCha_comp), ...])

    @staticmethod
    def process_sake_espirit(
        index,
        shared_ktraj,
        shared_kdata_shape,
        shared_kdata_dtype,
        shared_sensmaps_shape,
        shared_sensmaps_dtype,
        multi_contrast,
    ):
        # Reconnect to shared memory
        kdata = torch.frombuffer(
            shared_kdata_buffer,
            dtype=shared_kdata_dtype,
        ).reshape(shared_kdata_shape)

        sensmaps = torch.frombuffer(
            shared_sensmaps_buffer, dtype=shared_sensmaps_dtype
        ).reshape(shared_sensmaps_shape)

        if multi_contrast:
            sensmaps[:, :, :, :, index : index + 1, :, :] = sake_espirit_multicontrast(
                kdata[:, :, :, :, index : index + 1, :, :],
                shared_ktraj,
            )
        else:
            sensmaps[:, :, :, :, index : index + 1, :, :] = sake_espirit(
                kdata[:, :, :, :, index : index + 1, :, :],
                shared_ktraj,
            )

    def get_sensmaps(
        self,
        ktraj,
        sensmaps_shape,
        sensmaps_dtype=torch.complex64,
        buffer_dtype=ctypes.c_double,
        multi_contrast=False,
    ):
        NCha, NCol, NLin, NPar, NSli, NSet, NEco = self.kdata_shape

        # sensmaps_shape = (NCha, NImx, NImy, NPar, NSli, 1, 1)

        sensmaps_buffer = mp.RawArray(
            buffer_dtype,
            int(np.prod(sensmaps_shape)),
        )

        sensmaps = torch.frombuffer(
            sensmaps_buffer,
            dtype=sensmaps_dtype,
        ).reshape(sensmaps_shape)

        num_indices = NSli
        func = partial(
            KSpaceData.process_sake_espirit,
            shared_ktraj=ktraj,
            shared_kdata_shape=self.kdata_shape,
            shared_kdata_dtype=self.kdata_dtype,
            shared_sensmaps_shape=sensmaps_shape,
            shared_sensmaps_dtype=sensmaps_dtype,
            multi_contrast=multi_contrast,
        )

        KSpaceData.run_task(
            (None, self.kdata_buffer, sensmaps_buffer),
            func,
            num_indices,
        )

        return sensmaps
