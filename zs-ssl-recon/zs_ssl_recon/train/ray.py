import os
import sys

if os.getenv("ZS_SSL_RECON_SOFTWARE_DIR") is not None:
    sys.path.insert(0, os.getenv("ZS_SSL_RECON_SOFTWARE_DIR"))

from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from ..utils.parser import options_parser
from .train import train_loop_per_worker


def main():
    options = options_parser()

    scaling_config = ScalingConfig(
        num_workers=options["num_workers"],
        trainer_resources={"CPU": 1, "GPU": 0},
        resources_per_worker={
            "CPU": options["num_cpu_per_worker"],
            "GPU": options["num_gpu_per_worker"],
        },
        use_gpu=options["use_gpu"],
    )

    # Initialize the Trainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=options,
        scaling_config=scaling_config,
    )

    # Train the model.
    trainer.fit()


if __name__ == "__main__":
    main()
