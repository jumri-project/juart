import os
import subprocess

import torch
from threadpoolctl import threadpool_limits

from ..conopt.aux import pad_tensor
from ..conopt.aux.fourier import (
    fourier_transform_adjoint,
    fourier_transform_forward,
    nonuniform_fourier_transform_adjoint,
)
from ..conopt.tfs.fourier import nonuniform_transfer_function
from ..parim.autocalib import ac_region
from ..parim.espirit import espirit
from ..recon.sake import sake
from ..recon.sense import sense


def process_siemens_file(fname):
    # Define paths
    base_dir = f"/home/projects/qrage/sessions/{fname}"
    ismrmrd_dir = os.path.join(base_dir, "ismrmrd")

    # Create directories if they don't exist
    os.makedirs(ismrmrd_dir, exist_ok=True)

    # Set environment variable
    conda_lib_path = "/opt/conda/lib"
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{conda_lib_path}:{ld_library_path}"

    # Define input and output files
    input_file = os.path.join(base_dir, "twix", "mz_me_mpnrage3d.dat")
    output_file = os.path.join(ismrmrd_dir, "mz_me_mpnrage3d.h5")

    # Run the external command and stream the output
    command = ["siemens_to_ismrmrd", "-f", input_file, "-o", output_file]
    process = subprocess.run(command, check=True, stdout=None, stderr=None)

    return process


def sake_espirit(
    kdata,
    ktraj,
):
    import os

    os.environ["FFT_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"

    NAcl = 32
    scale = 1e-4
    NImx, NImy, ISet_coil, IEco_coil = 256, 256, slice(15, 19), slice(0, 1)
    NCha, NLin, NCol, NPar, NSli, NSet, NEco = kdata.shape

    with threadpool_limits(limits=1, user_api="blas"):
        k = ktraj[:, :, :, :, :, ISet_coil, IEco_coil]
        d = kdata[:, :, :, :, :, ISet_coil, IEco_coil] / scale

        k = k.reshape((2, -1, 1, 1, 1))
        d = d.reshape((NCha, 1, 1, -1, 1, 1, 1))

        kc, dc = ac_region(k, d, NAcl, NCol, ord=torch.inf)
        nKc = kc.shape[1]

        AHdc = nonuniform_fourier_transform_adjoint(
            kc, dc, (NAcl, NAcl, 1), (NCha, NAcl, NAcl, 1, 1, 1, 1)
        )
        Hc = nonuniform_transfer_function(
            kc, (NAcl, NAcl, 1, 1, 1, 1, nKc), oversampling=(2, 2, 1)
        )

        x = sake(AHdc, Hc, lamda_system=0.1, inner_iter=1, outer_iter=100)

        x = fourier_transform_forward(x, (1, 2))
        x = pad_tensor(x, (NCha, NImx, NImy, 1, 1, 1, 1))

        C = espirit(x[:, :, :, :, 0, :, 0], (NImx, NImy, 1))

    return C[:, :, :, :, None, None, None]


def espirit_sense(
    kspace_data_undersampled,
    partition_mask,
):
    # kspace_data_undersampled: (nC, nX, nY, nZ, nS, nTI, nTE)
    # partition_mask: (1, nX, 1, 1, 1, 1, 1)
    # C: (nC, nX, nY, nZ, nS)

    # pass to ESPIRiT: (nC, nX, nY, nZ, nTI * nTE)
    # although currently (nC, nX, nY, nZ, nTI)

    nC, nX, nY, nZ, nS, nTI, nTE = kspace_data_undersampled.shape

    scale = 1e-4

    kspace_data_undersampled = kspace_data_undersampled / scale

    # nS must be 1

    with threadpool_limits(limits=1, user_api="blas"):
        C_est = espirit(
            kspace_data_undersampled.reshape((nC, nX, nY, nZ, nTI * nTE)),
            (nX, nY, 1),
        ).reshape((nC, nX, nY, nZ, nS))

        AHd = torch.conj(C_est)[..., None, None] * fourier_transform_adjoint(
            kspace_data_undersampled, axes=(1, 2, 3)
        )
        AHd = torch.sum(AHd, dim=0)

        images = sense(
            C_est.to(torch.complex64),
            AHd.to(torch.complex64),
            partition_mask.to(torch.float32),
            lambda_wavelet=1e-4,
            channel_normalize=False,
            inner_iter=10,
            outer_iter=15,
        )

        kspaces_espirit = fourier_transform_forward(
            C_est[..., None, None] * images[None, ...], axes=(1, 2, 3)
        )

        kspaces_espirit = kspaces_espirit * scale

    return kspaces_espirit
