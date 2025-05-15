from types import SimpleNamespace

import numpy as np
import pypulseq as pp


def rasterize(
    time: float,
    raster: float,
    decimals: int = 9,
) -> float:
    """
    Rasterizes the given time to the nearest multiple of the raster time.

    Parameters:
    ----------
    time : float
        The input time to be rasterized.
    raster : float
        The time resolution to rasterize to.

    Returns:
    -------
    float
        The rasterized time.
    """
    return np.round(np.ceil(time / raster) * raster, decimals=decimals)


def set_grad(
    grad: SimpleNamespace,
    area: float,
    channel: str,
) -> SimpleNamespace:
    """
    Adjusts the gradient amplitude and assigns the channel.

    Parameters:
    ----------
    grad : SimpleNamespace
        The gradient object to be modified.
    area : float
        The desired gradient area.
    channel : str
        The channel along which the gradient is applied (e.g., 'x', 'y', 'z').

    Returns:
    -------
    SimpleNamespace
        The modified gradient object.
    """
    grad = pp.scale_grad(grad=grad, scale=area / grad.area)
    grad.channel = channel
    return grad
