import os

import yaml


def bool(value):
    if str(value).lower() in ["false", "0"]:
        return False

    else:
        return True


def getenv(type, key, delimiter=None):
    value = os.getenv(key)

    if value is None:
        return

    elif delimiter is None:
        return type(value)

    else:
        return [type(item.strip()) for item in value.split(delimiter)]


def env2dict():
    options = dict()

    options["disable_progress_bar"] = getenv(bool, "ZS_SSL_RECON_DISABLE_PROGRESS_BAR")
    options["timing_level"] = getenv(int, "ZS_SSL_RECON_TIMING_LEVEL")
    options["validation_level"] = getenv(int, "ZS_SSL_RECON_VALIDATION_LEVEL")

    options["save_checkpoint"] = getenv(bool, "ZS_SSL_RECON_SAVE_CHECKPOINT")
    options["checkpoint_frequency"] = getenv(int, "ZS_SSL_RECON_CHECKPOINT_FREQUENCY")

    options["load_model_state"] = getenv(bool, "ZS_SSL_RECON_LOAD_MODEL_STATE")
    options["load_averaged_model_state"] = getenv(
        bool, "ZS_SSL_RECON_LOAD_AVERAGED_MODEL_STATE"
    )
    options["load_optim_state"] = getenv(bool, "ZS_SSL_RECON_LOAD_OPTIM_STATE")
    options["load_metrics"] = getenv(bool, "ZS_SSL_RECON_LOAD_METRICS")

    # Can be variable for training and inference
    # Not advised to change during training
    # Requires implementation of gradient accumulation
    options["num_threads"] = getenv(int, "ZS_SSL_RECON_NUM_THREADS")
    options["num_cpu_per_worker"] = getenv(int, "ZS_SSL_RECON_NUM_CPU_PER_WORKER")
    options["num_gpu_per_worker"] = getenv(int, "ZS_SSL_RECON_NUM_GPU_PER_WORKER")
    options["num_workers"] = getenv(int, "ZS_SSL_RECON_NUM_WORKERS")

    options["use_gpu"] = getenv(bool, "ZS_SSL_RECON_USE_GPU")
    options["device"] = getenv(str, "ZS_SSL_RECON_DEVICE")

    # Fixed for training, variable for inference
    options["datasets"] = getenv(str, "ZS_SSL_RECON_DATASETS", delimiter=",")
    options["slices"] = getenv(int, "ZS_SSL_RECON_SLICES", delimiter=",")
    options["num_spokes"] = getenv(int, "ZS_SSL_RECON_NUM_SPOKES")
    options["model_tag"] = getenv(str, "ZS_SSL_RECON_MODEL_TAG")

    return options


def options_parser():
    # Read values from config file
    with open(
        os.getenv("ZS_SSL_RECON_CONFIG_FILE", default="../schemes/default.yaml"), "r"
    ) as file:
        options = yaml.safe_load(file)

    if isinstance(options["slices"], dict):
        options["slices"] = list(
            range(
                options["slices"]["start"],
                options["slices"]["stop"],
                options["slices"]["step"],
            )
        )

    # Overwrite values using environment variables
    options_env = env2dict()
    for key in options_env.keys():
        if options_env[key] is not None:
            options[key] = options_env[key]

    return options
