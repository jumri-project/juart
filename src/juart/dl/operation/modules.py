import torch

from ..utils.dist import gather_and_average_losses


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

            print("reading data")
            images_regridded = data["images_regridded"].to(device)
            kspace_trajectory = data["kspace_trajectory"].to(device)
            kspace_data = data["kspace_data"].to(device)
            kspace_mask_source = data["kspace_mask_source"].to(device)
            kspace_mask_target = data["kspace_mask_target"].to(device)
            sensitivity_maps = data["sensitivity_maps"].to(device)
            print("reading data done -> model initialization")
            # Forward path
            images_reconstructed = model(
                images_regridded,
                kspace_trajectory,
                kspace_mask=kspace_mask_source,
                sensitivity_maps=sensitivity_maps,
            )
            print("model initialization done -> loss fn initialization")
            # Loss
            loss = loss_fn(
                images_reconstructed,
                images_regridded,
                kspace_trajectory,
                kspace_data,
                kspace_mask_target,
                sensitivity_maps,
            )
            print("loss fn initialization done -> compute backward pass")
            # Backpropagation
            loss.backward()
            print("compute backward pass done -> compute accumulator")
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
        print("done with training()")

    return averaged_losses.tolist()


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

        # Forward path
        images_reconstructed = model(
            images_regridded,
            kspace_trajectory,
            sensitivity_maps=sensitivity_maps,
        )

    return images_reconstructed
