from typing import Tuple

import torch


def shift_forward(
    input_tensor: torch.Tensor,
    shift_number: Tuple[int, ...],
    shift_size: Tuple[int, ...],
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Applies forward shift operations to an input tensor along specified axes.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        The input tensor to which shifts are applied.
    shift_number : Tuple[int, ...]
        Number of shifts applied along each specified axis.
    shift_size : Tuple[int, ...]
        Size of each shift applied along the specified axes.
    axes : Tuple[int, ...]
        The axes along which the shifts are applied.

    Returns:
    -------
    output_tensor : torch.Tensor
        The shifted tensor with repeated shifts applied along the specified axes.
    """

    # Convert shift parameters to tensors
    shift_number = torch.tensor(shift_number, dtype=torch.int)
    shift_size = torch.tensor(shift_size, dtype=torch.int)

    # Total number of shifts
    N = torch.prod(shift_number)
    norm = torch.sqrt(N.float())

    # Repeat the input tensor N times along the first dimension
    output_tensor = torch.repeat_interleave(input_tensor, N, dim=0)

    # Apply shifts
    for index in range(N):
        shift = torch.tensor(
            torch.unravel_index(
                torch.tensor(index),
                shift_number.tolist(),
            )
        )
        output_tensor[index, ...] = torch.roll(
            output_tensor[index, ...],
            list(shift * shift_size),
            axes,
        )

    # Normalize the output tensor
    output_tensor = output_tensor / norm

    return output_tensor


def shift_adjoint(
    input_tensor: torch.Tensor,
    shift_number: Tuple[int, ...],
    shift_size: Tuple[int, ...],
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Applies the adjoint (inverse) shift operations to an input tensor along specified
    axes.

    Parameters:
    ----------
    input_tensor : torch.Tensor
        The input tensor to which the adjoint shifts are applied.
    shift_number : Tuple[int, ...]
        Number of shifts applied along each specified axis.
    shift_size : Tuple[int, ...]
        Size of each shift applied along the specified axes.
    axes : Tuple[int, ...]
        The axes along which the adjoint shifts are applied.

    Returns:
    -------
    output_tensor : torch.Tensor
        The tensor after applying the adjoint shifts and summing over the shifted
        dimension.
    """

    # Convert shift parameters to tensors
    shift_number = torch.tensor(shift_number, dtype=torch.int)
    shift_size = torch.tensor(shift_size, dtype=torch.int)

    # Total number of shifts
    N = torch.prod(shift_number)
    norm = torch.sqrt(N.float())

    # Clone the input tensor to prepare for adjoint operation
    output_tensor = input_tensor.clone()

    # Apply adjoint shifts
    for index in range(N):
        shift = torch.tensor(
            torch.unravel_index(
                torch.tensor(index),
                shift_number.tolist(),
            )
        )
        output_tensor[index, ...] = torch.roll(
            output_tensor[index, ...],
            list(-shift * shift_size),
            axes,
        )

    # Sum along the shifted dimension and normalize
    output_tensor = torch.sum(output_tensor, dim=0, keepdims=True)
    output_tensor = output_tensor / norm

    return output_tensor
