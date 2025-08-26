import os
import sys

if os.getenv("ZS_SSL_RECON_SOFTWARE_DIR") is not None:
    sys.path.insert(0, os.getenv("ZS_SSL_RECON_SOFTWARE_DIR"))


import numpy as np
import ray
from torch.optim.swa_utils import AveragedModel

from ..checkpoint.manager import CheckpointManager
from ..data.inference import DatasetInference, ImageStore
from ..model.unrollnet import SingleContrastUnrolledNet, UnrolledNet
from ..operation.modules import inference
from ..utils.parser import options_parser


@ray.remote
class InferenceActor:
    def __init__(self, options, checkpoint, device=None):
        nX, nY, nTI, nTE = options["shape"]

        if options["groups"] == 1:
            self.model = UnrolledNet(
                (nX, nY),
                contrasts=nTI * nTE,
                features=options["features"],
                CG_Iter=options["CG_Iter"],
                num_unroll_blocks=options["num_unroll_blocks"],
                activation=options["activation"],
                phase_normalization=options["phase_normalization"],
                disable_progress_bar=options["disable_progress_bar"],
                device=device,
            )
        else:
            self.model = SingleContrastUnrolledNet(
                (nX, nY),
                contrasts=nTI * nTE,
                features=options["features"],
                CG_Iter=options["CG_Iter"],
                num_unroll_blocks=options["num_unroll_blocks"],
                activation=options["activation"],
                phase_normalization=options["phase_normalization"],
                disable_progress_bar=options["disable_progress_bar"],
                device=device,
            )

        if options["load_averaged_model_state"]:
            self.model = AveragedModel(self.model)
            self.model.load_state_dict(checkpoint["averaged_model_state"])
        else:
            self.model.load_state_dict(checkpoint["model_state"])

        self.dataset = DatasetInference(
            options["session_dir"],
            options["datasets"],
            options["slices"],
            options["num_spokes"],
            endpoint_url=options["endpoint_url"],
            backend=options["data_backend"],
            device=device,
        )

        self.store = ImageStore(
            options["image_dir"],
            options["datasets"],
            options["slices"],
            options["model_tag"],
            endpoint_url=options["endpoint_url"],
            backend=options["image_backend"],
        )

        self.device = device

    def inference(self, index):
        images = inference(self.dataset[index], self.model, device=self.device)

        self.store.save(images, index)


def main():
    options = options_parser()

    if options["use_gpu"]:
        device = "cuda"
    else:
        device = "cpu"

    store = ImageStore(
        options["image_dir"],
        options["datasets"],
        options["slices"],
        options["model_tag"],
        endpoint_url=options["endpoint_url"],
        backend=options["image_backend"],
    )

    for dataset_index, dataset in enumerate(options["datasets"]):
        store.create(dataset_index, overwrite=True)

    checkpoint_manager = CheckpointManager(
        options["model_dir"] + options["model_tag"],
        endpoint_url=options["endpoint_url"],
        backend=options["model_backend"],
    )

    # Averaged model has priority
    if options["load_averaged_model_state"]:
        checkpoint = checkpoint_manager.load(
            ["averaged_model_state", "iteration"], map_location="cpu"
        )
        iteration = checkpoint["iteration"]
        print(f"Loaded averaged at iteration {iteration}.")
    elif options["load_model_state"]:
        checkpoint = checkpoint_manager.load(
            ["model_state", "iteration"], map_location="cpu"
        )
        iteration = checkpoint["iteration"]
        print(f"Loaded model at iteration {iteration}.")
    else:
        print("Either load_model_state or load_averaged_model_state must be true.")

    # Create actors
    inference_actors = list()
    for _ in range(options["num_workers"]):
        inference_actors.append(
            InferenceActor.options(
                num_cpus=options["num_cpu_per_worker"],
                num_gpus=options["num_gpu_per_worker"],
            ).remote(options, checkpoint, device=device)
        )
    print(f"Created {len(inference_actors)} actors")

    # Run inference
    results = list()
    for dataset_index, dataset in enumerate(options["datasets"]):
        for slice_index, slc in enumerate(options["slices"]):
            index = len(options["slices"]) * dataset_index + slice_index
            actor_index = np.mod(
                index // (len(options["slices"]) // len(inference_actors)),
                len(inference_actors),
            )
            results.append(
                inference_actors[actor_index].inference.remote(
                    index,
                )
            )
            print(
                (
                    "Scheduled inference for "
                    + f"dataset {options['datasets'][dataset_index]} - "
                    + f"slice {options['slices'][slice_index]} on "
                    + f"actor {actor_index} ..."
                )
            )

    # Wait until results are ready
    ray.get(results)
