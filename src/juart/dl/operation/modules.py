import torch

from torch import distributed as dist
from ..utils.dist import gather_and_average_losses
import torch.profiler
import sys
sys.path.insert(0, "../../src")
from juart.conopt.functional.__init__ import pad_tensor, crop_tensor


def training(
    indices,
    dataset,
    model,
    loss_fn,
    optimizer,
    accumulator,
    group=None,
    device=None,
):
    losses = []

    model.train()

    with torch.autograd.graph.save_on_cpu():
        for index in indices:
            # Prepare data
            data = dataset[index]

            print(f"Rank {dist.get_rank()} - reading data")

            images_regridded = data["images_regridded"].to(device)
            kspace_trajectory = data["kspace_trajectory"].to(device)
            kspace_data = data["kspace_data"].to(device)
            kspace_mask_source = data["kspace_mask_source"].to(device)
            kspace_mask_target = data["kspace_mask_target"].to(device)
            sensitivity_maps = data["sensitivity_maps"].to(device)
            print(f"Rank {dist.get_rank()} - reading data done -> model initialization")
            # Forward path:
            dist.barrier()

            if model.net_structure == "UNet":
                corr = int(2**torch.ceil(torch.log2(torch.Tensor([images_regridded.shape[0]]))).item())
                images_regridded = pad_tensor(images_regridded,
                                              (corr,
                                               corr,
                                               corr,
                                               images_regridded.shape[3],
                                               images_regridded.shape[4])
                                             )

                sensitivity_maps = pad_tensor(sensitivity_maps,
                                              (sensitivity_maps.shape[0],
                                               corr,
                                               corr,
                                               corr)
                                             )

                if model.pad_to != 0:
                    pad_to = model.pad_to
                    if len(model.kernel_size) == 2:
                        images_regridded = crop_tensor(images_regridded,(pad_to,pad_to,images_regridded.shape[2],images_regridded.shape[3],images_regridded.shape[4]))
                        sensitivity_maps = crop_tensor(sensitivity_maps,(sensitivity_maps.shape[0],pad_to,pad_to,images_regridded.shape[2]))

                    elif len(model.kernel_size) == 3:
                        images_regridded = crop_tensor(images_regridded,(pad_to,pad_to,pad_to,images_regridded.shape[3],images_regridded.shape[4]))
                        sensitivity_maps = crop_tensor(sensitivity_maps,(sensitivity_maps.shape[0],pad_to,pad_to,pad_to))

            images_reconstructed = model(
                images_regridded,
                kspace_trajectory,
                kspace_mask=kspace_mask_source,
                sensitivity_maps=sensitivity_maps,
            )

            print(f"Rank {dist.get_rank()} - model initialization done -> loss fn initialization")
            dist.barrier()
            # Loss
            loss = loss_fn(
                images_reconstructed,
                images_regridded,
                kspace_trajectory,
                kspace_data,
                kspace_mask_target,
                sensitivity_maps,
            )  
            print(f"Rank {dist.get_rank()} - loss fn initialization done -> compute backward pass")
            dist.barrier() 
            # Backpropagation
            loss.backward()  
            print(f"Rank {dist.get_rank()} - compute backward pass done -> compute accumulator")
            dist.barrier() 
            # Accumulate gradients
            accumulator.accumulate()

            losses.append(loss.item())

        accumulator.apply()

        optimizer.step()
        optimizer.zero_grad()

        # Average loss
        averaged_losses = gather_and_average_losses(
            torch.tensor(losses), group=group, device=device
        )

    #return averaged_losses.tolist(), images_reconstructed
    return averaged_losses

def validation(
    indices,
    dataset,
    model,
    loss_fn,
    group=None,
    device=None,
):
    model.eval()

    with torch.no_grad():
        for index in indices:
            # Prepare data
            data = dataset[index]

            images_regridded = data["images_regridded"].to(device)
            kspace_trajectory = data["kspace_trajectory"].to(device)
            kspace_data = data["kspace_data"].to(device)
            kspace_mask_source = data["kspace_mask_source"].to(device)
            kspace_mask_target = data["kspace_mask_target"].to(device)
            sensitivity_maps = data["sensitivity_maps"].to(device)

            # Forward path
            images_reconstructed = model(
                images_regridded,
                kspace_trajectory,
                kspace_mask=kspace_mask_source,
                sensitivity_maps=sensitivity_maps,
            )

            # Loss
            loss = loss_fn(
                images_reconstructed,
                images_regridded,
                kspace_trajectory,
                kspace_data,
                kspace_mask_target,
                sensitivity_maps,
            )

        # Average loss
        averaged_losses = gather_and_average_losses(
            torch.tensor(losses),
            group=group,
            device=device,
        )

    return averaged_losses.tolist()


def inference(
    data,
    model,
    device=None,
):
    model.eval()

    with torch.no_grad():
        images_regridded = data["images_regridded"].to(device)
        kspace_trajectory = data["kspace_trajectory"].to(device)
        sensitivity_maps = data["sensitivity_maps"].to(device)

        if model.module.net_structure == "UNet":
            corr = int(2**torch.ceil(torch.log2(torch.Tensor([images_regridded.shape[0]]))).item())
            images_regridded = pad_tensor(images_regridded,
                                          (corr,
                                           corr,
                                           corr,
                                           images_regridded.shape[3],
                                           images_regridded.shape[4])
                                         )

            sensitivity_maps = pad_tensor(sensitivity_maps,
                                          (sensitivity_maps.shape[0],
                                           corr,
                                           corr,
                                           corr)
                                         )

            if model.module.pad_to != 0:
                pad_to = model.module.pad_to
                if len(model.module.kernel_size) == 2:
                    images_regridded = crop_tensor(images_regridded,(pad_to,pad_to,images_regridded.shape[2],images_regridded.shape[3],images_regridded.shape[4]))
                    sensitivity_maps = crop_tensor(sensitivity_maps,(sensitivity_maps.shape[0],pad_to,pad_to,images_regridded.shape[2]))

                elif len(model.module.kernel_size) == 3:
                    images_regridded = crop_tensor(images_regridded,(pad_to,pad_to,pad_to,images_regridded.shape[3],images_regridded.shape[4]))
                    sensitivity_maps = crop_tensor(sensitivity_maps,(sensitivity_maps.shape[0],pad_to,pad_to,pad_to))

        # Forward path
        images_reconstructed = model(
            images_regridded,
            kspace_trajectory,
            sensitivity_maps=sensitivity_maps,
        )

    return images_reconstructed
