import torch
import numpy as np


def ema(
    data: torch.Tensor,
    alpha: float = 0.0,
    tau: float = 0.0,
):
    '''
    returns exponential smoothed version of the data.

    Parameter:
    data - torch.Tensor, shape [n,]
        data that should be smoothed.
    alpha - float = 0.0
        Parameter that describes the decay of the exponential function.
    tau - float = 0.0
        Parameter that describes how many previous values should be noticibly 
        respected in the calculation of the current value

    If alpha is already given, then tau is not necessary.
    Tau is just necessary if you want to compute alpha automatically.
    Either alpha or tau must be != 0.
    When alpha != 0, tau will be ignored.
    '''

    if alpha == 0.0 and tau == 0.0:
        raise ValueError("invalid input combination of alpha and tau. One of them must be != 0")

    if alpha == 0.0:
        alpha = 1 - np.exp(-1/tau)

    s = [data[0]]

    for y in data:
        s.append(alpha*y + (1-alpha)*s[-1])

    return s


def ema_dict(
    data: dict,
    alpha: float = 0.0,
    tau: float = 0.0,
):
    '''
    returns exponential smoothed version of the data.

    Parameter:
    data - torch.Tensor, shape [n,]
        data that should be smoothed.
    alpha - float = 0.0
        Parameter that describes the decay of the exponential function.
    tau - float = 0.0
        Parameter that describes how many previous values should be noticibly 
        respected in the calculation of the current value

    If alpha is already given, then tau is not necessary.
    Tau is just necessary if you want to compute alpha automatically.
    Either alpha or tau must be != 0.
    When alpha != 0, tau will be ignored.
    '''

    if alpha == 0.0 and tau == 0.0:
        raise ValueError("invalid input combination of alpha and tau. One of them must be != 0")

    if alpha == 0.0:
        alpha = 1 - np.exp(-1/tau)

    s = dict()
    for key in data.keys():
        s[key] = [data[key][0]]
        for y in data[key]:
            s[key].append(alpha*y + (1-alpha)*s[key][-1])

    return s