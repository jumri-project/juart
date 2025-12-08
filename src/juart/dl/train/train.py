import gc
import time

import numpy as np
import torch
import torch.distributed as dist

from ..checkpoint.manager import CheckpointManager
from ..data.training import DatasetTraining
from ..loss.loss import JointLoss
from ..model.unrollnet import (
    ExponentialMovingAverageModel,
    LookaheadModel,
    SingleContrastUnrolledNet,
    UnrolledNet,
)
from ..operation.modules import training, validation
from ..utils.dist import GradientAccumulator


def shuffled_indices(num_samples, num_epochs, rng):
    indices = np.repeat(np.arange(num_samples), num_epochs)
    indices = indices.reshape((num_samples, num_epochs))
    indices = rng.permuted(indices, axis=0)
    indices = indices.T.ravel()

    # Check if each sample is used once and only once in every epoch
    assert indices.size == num_samples * num_epochs
    for i in np.split(indices, num_epochs):
        assert np.unique(i).size == num_samples

    return indices


def train_loop_per_worker(options):
    # Fix random seed
    np.random.seed(0)
    torch.manual_seed(0)

    torch.set_num_threads(options["num_threads"])
    torch.set_num_interop_threads(options["num_threads"])

    # print(torch.__config__.show())
    # print(torch.__config__.parallel_info())

    global_rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())

    print(f"Rank {global_rank} - Intialize local groups ...")
    dist.barrier()

    group_size = options["group_size"]

    for rank in range(0, world_size, group_size):
        ranks = list(range(rank, rank + group_size, 1))
        if global_rank in ranks:
            print(f"Rank {global_rank} is in group {ranks} ...")
            group = dist.new_group(ranks, backend="gloo")
        dist.barrier()

    group_rank = dist.get_group_rank(group, global_rank)
    group_index = global_rank // group_size
    num_groups = world_size // group_size

    print(f"Rank {global_rank} is local rank {group_rank} ...")
    dist.barrier()

    if options["use_gpu"] and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device_rank = np.mod(global_rank, torch.cuda.device_count())
        device = f"cuda:{device_rank}"
        print(
            f"Rank {global_rank} - Using CUDA device {device_rank} of {num_devices} ..."
        )
    else:
        device = "cpu"

    print(f"Rank {global_rank} is using device {device} ...")
    dist.barrier()

    nD = len(options["datasets"])
    nS = len(options["slices"])
    shape = tuple(options["shape"])

    num_epochs = options["epochs"]

    # The number of batches that are computed serially via gradient accumulation
    batch_size = options["batch_size"]
    batch_size_local = batch_size // num_groups

    num_iterations = nD * nS * num_epochs

    rng = np.random.default_rng(seed=0)

    training_indices = shuffled_indices(nD * nS, num_epochs, rng)
    training_indices_batched = training_indices.reshape(
        (-1, batch_size_local, num_groups)
    )

    validation_indices = shuffled_indices(nD * nS, num_epochs, rng)
    validation_indices_batched = validation_indices.reshape(
        (-1, batch_size_local, num_groups)
    )

    # Prepare models and optimizer

    if options["groups"] == 1:
        model = UnrolledNet(
            shape,
            features=options["features"],
            CG_Iter=options["CG_Iter"],
            num_unroll_blocks=options["num_unroll_blocks"],
            activation=options["activation"],
            disable_progress_bar=options["disable_progress_bar"],
            timing_level=options["timing_level"],
            validation_level=options["validation_level"],
            device=device,
        )
    else:
        model = SingleContrastUnrolledNet(
            shape,
            features=options["features"],
            CG_Iter=options["CG_Iter"],
            num_unroll_blocks=options["num_unroll_blocks"],
            activation=options["activation"],
            disable_progress_bar=options["disable_progress_bar"],
            timing_level=options["timing_level"],
            validation_level=options["validation_level"],
            device=device,
        )
    loss_fn = JointLoss(
        (3, 3),
        weights_kspace_loss=options["weight_kspace_loss"],
        weights_ispace_loss=options["weight_ispace_loss"],
        weights_wavelet_loss=options["weight_wavelet_loss"],
        weights_hankel_loss=options["weight_hankel_loss"],
        weights_casorati_loss=options["weight_casorati_loss"],
        dim_kspace_loss=options["dim_kspace_loss"],
        dim_ispace_loss=options["dim_ispace_loss"],
        dim_wavelet_loss=options["dim_wavelet_loss"],
        normalized_loss=options["normalized_loss"],
        timing_level=options["timing_level"],
        validation_level=options["validation_level"],
        group=group,
        device=device,
    )
    if options["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=options["lr"],
            betas=options["betas"],
            eps=options["eps"],
            weight_decay=options["weight_decay"],
        )
    elif options["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=options["lr"],
            betas=options["betas"],
            eps=options["eps"],
            weight_decay=options["weight_decay"],
        )
    elif options["optimizer"] == "RAdam":
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=options["lr"],
            betas=options["betas"],
            eps=options["eps"],
            weight_decay=options["weight_decay"],
        )

    accumulator = GradientAccumulator(
        model,
        accumulation_steps=batch_size_local,
        max_norm=1.0,
        normalized_gradient=options["normalized_gradient"],
    )

    # model = DDP(model)

    if options["averaged_model"] == "EMA":
        print(f"Rank {global_rank} - ExponentialMovingAverageModel")
        averaged_model = ExponentialMovingAverageModel(
            model,
            decay=options["ema_decay"],
        )
    elif options["averaged_model"] == "Lookahead":
        print(f"Rank {global_rank} - LookaheadModel")
        averaged_model = LookaheadModel(
            model,
            alpha=options["lookahead_alpha"],
            k=options["lookahead_k"],
        )

    checkpoint_manager = CheckpointManager(
        options["model_dir"],
        root_dir=options["root_dir"],
        endpoint_url=options["endpoint_url"],
        backend=options["model_backend"],
    )

    if options["load_model_state"]:
        print(f"Rank {global_rank} - Loading model state ...")
        checkpoint = checkpoint_manager.load(["model_state"], map_location=device)
        if all(checkpoint.values()):
            model.load_state_dict(checkpoint["model_state"])
        else:
            print(f"Rank {global_rank} - Could not load model state.")

    if options["load_averaged_model_state"]:
        print(f"Rank {global_rank} - Loading averaged model state ...")
        checkpoint = checkpoint_manager.load(
            ["averaged_model_state"], map_location=device
        )
        if all(checkpoint.values()):
            averaged_model.load_state_dict(checkpoint["averaged_model_state"])
        else:
            print(f"Rank {global_rank} - Could not load averaged model state.")

    if options["load_optim_state"]:
        print(f"Rank {global_rank} - Loading optim state ...")
        checkpoint = checkpoint_manager.load(["optim_state"], map_location=device)
        if all(checkpoint.values()):
            optimizer.load_state_dict(checkpoint["optim_state"])
        else:
            print(f"Rank {global_rank} - Could not load optim state.")

    total_trn_loss = list()
    total_val_loss = list()
    iteration = 0

    if options["load_metrics"]:
        print(f"Rank {global_rank} - Loading metrics ...")
        checkpoint = checkpoint_manager.load(["trn_loss", "val_loss", "iteration"])
        if all(checkpoint.values()):
            total_trn_loss = list(checkpoint["trn_loss"])
            total_val_loss = list(checkpoint["val_loss"])
            iteration = checkpoint["iteration"]
        else:
            print(f"Rank {global_rank} - Could not load metrics.")

    print(f"Rank {global_rank} - Continue with iteration {iteration} ...")

    training_data = DatasetTraining(
        options["data_dir"],
        options["datasets"],
        options["slices"],
        options["num_spokes"],
        options["fractions"],
        mode="training",
        group_rank=group_rank,
        root_dir=options["root_dir"],
        endpoint_url=options["endpoint_url"],
        backend=options["data_backend"],
    )
    validation_data = DatasetTraining(
        options["data_dir"],
        options["datasets"],
        options["slices"],
        options["num_spokes"],
        options["fractions"],
        mode="validation",
        group_rank=group_rank,
        root_dir=options["root_dir"],
        endpoint_url=options["endpoint_url"],
        backend=options["data_backend"],
    )

    # Train model
    while iteration < num_iterations:
        tic = time.time()

        # Reset the seed so that training can be resumed
        np.random.seed(iteration)
        torch.manual_seed(iteration)

        training_index = training_indices_batched[
            iteration // batch_size,
            :,
            group_index,
        ].tolist()
        validation_index = validation_indices_batched[
            iteration // batch_size, :, group_index
        ].tolist()

        if options["model_training"]:
            print(f"Rank {global_rank} - Training index {training_index} ...")

            trn_loss = training(
                training_index,
                training_data,
                model,
                loss_fn,
                optimizer,
                accumulator,
                group=group,
                device=device,
            )

            averaged_model.update_parameters(
                model,
            )

            torch.cuda.empty_cache()
            gc.collect()

        else:
            trn_loss = [0] * batch_size

        if options["model_validation"]:
            print(f"Rank {global_rank} - Validation index {validation_index} ...")

            val_loss = validation(
                validation_index,
                validation_data,
                averaged_model,
                loss_fn,
                group=group,
                device=device,
            )
            torch.cuda.empty_cache()
            gc.collect()

        else:
            val_loss = [0] * batch_size

        total_trn_loss += trn_loss
        total_val_loss += val_loss

        if global_rank == 0:
            # Completed epoch
            if (
                options["save_checkpoint"]
                and np.mod(iteration + batch_size, nD * nS) == 0
            ):
                print("Creating tagged checkpoint ...")

                checkpoint = {
                    "iteration": iteration + batch_size,
                    "model_state": model.state_dict(),
                    "averaged_model_state": averaged_model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "trn_loss": total_trn_loss,
                    "val_loss": total_val_loss,
                }

                epoch = (iteration + batch_size) // (nD * nS)
                checkpoint_manager.save(checkpoint, tag=f"_epoch_{epoch}")

                if options["single_epoch"]:
                    # Also save the checkpoint as untagged checkpoint
                    # Otherwise, training will be stuck in endless loop
                    checkpoint_manager.save(checkpoint)
                    checkpoint_manager.release()
                    break

            # Intermediate checkpoint
            elif (
                options["save_checkpoint"]
                and np.mod(iteration + batch_size, options["checkpoint_frequency"]) == 0
            ):
                print("Creating untagged checkpoint ...")

                checkpoint = {
                    "iteration": iteration + batch_size,
                    "model_state": model.state_dict(),
                    "averaged_model_state": averaged_model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "trn_loss": total_trn_loss,
                    "val_loss": total_val_loss,
                }

                checkpoint_manager.save(checkpoint, block=False)

            toc = time.time() - tic

            print(
                (
                    f"Iteration: {iteration} - "
                    + f"Elapsed time: {toc:.0f} - "
                    + f"Training loss: {[f'{loss:.3f}' for loss in trn_loss]} - "
                    + f"Validation loss: {[f'{loss:.3f}' for loss in val_loss]}"
                )
            )

        torch.cuda.empty_cache()
        gc.collect()

        iteration += batch_size
