from typing import Literal

import torch


def resize(
    input: torch.Tensor,
    size: tuple[int, ...] | list[int] | int,
    dims: tuple[int, ...] | list[int] | int,
    mode: Literal["centric", "left", "rigth"] = "centric",
) -> torch.Tensor:
    # Ensure dims and size are iterable
    if isinstance(dims, int):
        dims = [dims]
    if isinstance(size, int):
        size = [size]
    for d, s in zip(dims, size):
        curr_size = input.shape[d]
        pad_left = pad_right = 0
        if mode == "centric":
            total_pad = s - curr_size
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
        elif mode == "left":
            pad_left = s - curr_size
        elif mode == "rigth":
            pad_right = s - curr_size
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'centric', 'left', or 'rigth'."
            )

        if pad_left < 0 or pad_right < 0:
            # Crop if new size is smaller
            start = -pad_left if pad_left < 0 else 0
            end = curr_size + pad_right if pad_right < 0 else curr_size
            slices = [slice(None)] * input.ndim
            slices[d] = slice(start, end)
            input = input[tuple(slices)]
        else:
            pad_width = [(0, 0)] * input.ndim
            pad_width[d] = (pad_left, pad_right)
            input = torch.nn.functional.pad(
                input, [p for pair in reversed(pad_width) for p in pair], value=0
            )

    return input
