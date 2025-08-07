import io
import os

import h5py
from s3fs import S3FileSystem


def prepare_dataset(
    subjects=["7T1026", "7T1027", "7T1028", "7T1029"],
    endpoint_url="https://s3.fz-juelich.de",
):
    slices = range(160)

    os.environ["AWS_ACCESS_KEY_ID"] = "21Y653Y987AAHCYC4HD7"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "hiFxA0bqld0SUDfRsrHJW9aCiJAAszh3UStyhLlv"

    fs = S3FileSystem(anon=False, endpoint_url=endpoint_url)

    for subject in subjects:
        source_file = f"qrage/sessions/{subject}/preproc/mz_me_mpnrage3d_grappa.h5"

        for iS in slices:
            print(subject, iS)

            target_file = (
                f"qrage/datasets/{subject}/preproc/mz_me_mpnrage3d_grappa_{iS:03d}.h5"
            )

            with h5py.File(fs.open(source_file, mode="rb")) as f:
                sensitivity_maps = f["C"][:, :, :, :, iS : iS + 1, :, :]
                kspace_trajectory = f["k"][:, :, :, :, :, :, :]
                kspace_data = f["d"][:, :, :, :, iS : iS + 1, :, :]

            with io.BytesIO() as buffer:
                with h5py.File(buffer, "a") as f:
                    f.create_dataset("C", data=sensitivity_maps)
                    f.create_dataset("k", data=kspace_trajectory)
                    f.create_dataset("d", data=kspace_data)

                with fs.open(target_file, "wb") as f:
                    f.write(buffer.getvalue())
