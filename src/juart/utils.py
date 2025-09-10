import time
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Literal, Optional

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


def verbose_print(current_level: int, trigger_level: int, msg: str, *args):
    """
    Print a message if current_level >= trigger_level.

    Example:
        verbose_print(verbose, 3, "Shape: %s", x.shape)
    """
    if current_level >= trigger_level:
        if args:
            print(msg % args, sep="")
        else:
            print(msg, sep="")


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    timers: ClassVar[dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} ms"
    current_level: int = 3
    trigger_level: int = 3
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time

        elapsed_time *= 1000  # Convert to milliseconds

        self._start_time = None

        # Report elapsed time
        if self.logger and self.current_level >= self.trigger_level:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time
