import torch
import torch.distributed as dist
import torch.nn as nn


def gather_and_average_losses(
    losses: torch.Tensor, group=None, device=None
) -> torch.Tensor:
    """
    Gathers and averages losses across groups using a 1D tensor.

    Parameters:
    - local_loss (torch.Tensor): The local loss for each worker, a single-element tensor.
    - group (torch.distributed.ProcessGroup): The group of workers processing the same dataset.
    - device (torch.device): Device on which the tensors are allocated.

    Returns:
    - averaged_losses (torch.Tensor): A 1D tensor containing the averaged loss for each group, available on all workers.
    """

    group_size = dist.get_world_size(group=group)
    world_size = dist.get_world_size()

    # Determine this worker's group index and calculate number of groups
    group_index = dist.get_rank() // group_size
    num_groups = world_size // group_size
    accumulation_steps = losses.shape[0]

    # Initialize a 2D tensor to hold losses, with one entry per group
    averaged_losses = torch.zeros((accumulation_steps, num_groups), device=device)

    # Place the local loss in the appropriate location in the 2D losses tensor
    averaged_losses[:, group_index] = losses

    # Perform all_reduce to sum all losses across all workers
    dist.all_reduce(averaged_losses, op=dist.ReduceOp.SUM)

    # Average the losses by dividing by group_size
    averaged_losses = averaged_losses / group_size

    averaged_losses = averaged_losses.ravel()

    return averaged_losses


def average_images(
    images: torch.Tensor,
    group=None,
) -> torch.Tensor:
    images = images.clone()
    images[~torch.isfinite(images)] = 0
    images = images.contiguous()
    images /= len(dist.get_process_group_ranks(group))
    dist.all_reduce(images, op=dist.ReduceOp.SUM, group=group)

    return images


class GradientAccumulator:
    def __init__(
        self,
        model,
        accumulation_steps=1,
        max_norm=None,
        norm_type=2.0,
        normalized_gradient=False,
        group=None,
    ):
        """
        Initializes the Gradient Accumulator.

        Parameters:
        - model (torch.nn.Module): The model whose gradients will be accumulated.
        - accumulation_steps (int): Number of steps over which to accumulate gradients.
        - max_norm (float): Maximum norm for gradient clipping.
        - norm_type (float): Type of the p-norm for gradient clipping (default: 2.0).
        - group (torch.distributed.ProcessGroup, optional): Process group to use for distributed averaging.
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.accumulation_index = 0
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.normalized_gradient = normalized_gradient
        self.group = group

        # Initialize accumulated gradients
        self.accumulated_grads = [torch.zeros_like(p) for p in model.parameters()]

    def accumulate(self):
        """
        Accumulates gradients, setting NaNs and Infs to zero.
        """

        # Measure gradient norm
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float("inf"),
            norm_type=self.norm_type,
        )
        print(
            f"Rank {dist.get_rank()} - Index {self.accumulation_index} - Gradient norm: {gradient_norm.item():.3f}"
        )

        # Normalize gradient
        if self.normalized_gradient:
            gradient_scaling = 1 / gradient_norm
        else:
            gradient_scaling = 1

        # Add model gradients to accumulated gradients and zero model gradients afterwards
        for param, accumulated_grad in zip(
            self.model.parameters(), self.accumulated_grads
        ):
            if param.grad is not None:
                param.grad[~torch.isfinite(param.grad)] = 0
                accumulated_grad.add_(gradient_scaling * param.grad)
                param.grad.zero_()

        self.accumulation_index += 1

    def apply(self):
        """
        Averages the accumulated gradients and applies gradient clipping if max_norm is set.
        Replaces the model's gradients with the accumulated gradients.
        Resets the accumulated gradients and step counter.
        """

        # Synchronize and average across accumulation steps and distributed workers
        for grad in self.accumulated_grads:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.group)
            grad /= self.accumulation_steps
            grad /= dist.get_world_size()

        # Apply accumulated gradients to model and zero accumulated gradients afterwards
        for param, accumulated_grad in zip(
            self.model.parameters(), self.accumulated_grads
        ):
            if param.grad is not None:
                param.grad.copy_(accumulated_grad)
                accumulated_grad.zero_()

        # Apply gradient clipping if specified
        # Note: nn.utils.clip_grad_norm_ works only on model.parameters()
        # but not on accumulated_grads, because accumulated_grads
        # is just a list of tensors, without any .grad objects
        if self.max_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_norm,
                norm_type=self.norm_type,
            )

        # Measure averaged gradient norm
        averaged_gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float("inf"),
            norm_type=self.norm_type,
        )
        if dist.get_rank() == 0:
            print(f"Averaged gradient norm: {averaged_gradient_norm.item():.3f}")

        self.accumulation_index = 0
