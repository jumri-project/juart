import torch

from ..conopt.functional import crop_tensor, pad_tensor
from ..conopt.functional.fourier import (
    fourier_transform_adjoint,
    fourier_transform_forward,
)


def interpolation(
    x: torch.Tensor,
    baseresolution: int = 256,
    Npad: int = 100,
    beta: float = 4.0,
    axes: int = 1,
    readoutDownsampling: bool = False,
) -> torch.Tensor:
    """
    Interpolation of the radial spokes.

    Args:
        x (torch.Tensor): Data to be interpolated.
        baseresolution (int): Base resolution.
        Npad (int): Padding on the edges of the k-space for interpolation.
        beta (float): Central part that is considered to find intersection points.
        axes (int): Axes for Fourier transform.
        readoutDownsampling (bool): Whether to apply readout downsampling.

    Returns:
        torch.Tensor: Interpolated data.
    """
    nC = x.shape[0]
    x = fourier_transform_adjoint(x, axes=axes)

    if not readoutDownsampling:
        freq = torch.linspace(-0.5, 0.5, steps=baseresolution, dtype=torch.float64)
        x = x * torch.exp(1j * torch.pi * freq[None, :, None])

    x = pad_tensor(x, (nC, Npad * baseresolution, 2))
    x = fourier_transform_forward(x, axes=axes)
    x = crop_tensor(x, (nC, int(Npad * beta), 2))

    return x


def differentiation(
    data_interpol: torch.Tensor,
) -> torch.Tensor:
    """
    Performs differentiation on interpolated data.

    Args:
        data_interpol (torch.Tensor): Interpolated k-space data.

    Returns:
        torch.Tensor: Differentiated data.
    """
    diff = data_interpol[..., 0, None].swapaxes(-1, -2) - data_interpol[..., 1, None]
    diff = torch.sum(torch.abs(diff) ** 2, axis=0)

    return diff


def indexing(
    diff: torch.Tensor,
    Npad: int = 100,
    beta: float = 4.0,
) -> torch.Tensor:
    """
    Find the index of the minimum value in diff and normalize.

    Args:
        diff (torch.Tensor): Differentiated data.
        Npad (int): Padding size.
        beta (float): Scaling factor.

    Returns:
        torch.Tensor: Normalized index of the minimum value.
    """
    ind_cross = torch.unravel_index(torch.argmin(torch.abs(diff)), diff.shape)
    ind_cross = torch.tensor(ind_cross, dtype=torch.float64)

    # Normalize the crossing indices
    ind_cross = (ind_cross - Npad * beta / 2) / Npad

    return ind_cross


def A_matrix(
    angles: torch.Tensor,
    n: int,
    nDiff: int = 17,
) -> torch.Tensor:
    """
    Constructs the matrix A for gradient delay estimation.

    Args:
        angles (torch.Tensor): Array of angles.
        n (int): Current index.
        nDiff (int): Index difference.

    Returns:
        torch.Tensor: 2x3 matrix A.
    """
    angle0 = angles[n + nDiff]
    angle1 = angles[n]

    ksi1 = torch.cos(angle1) - torch.cos(angle0)
    ksi2 = torch.sin(angle1) - torch.sin(angle0)

    return torch.tensor([[ksi1, 0, ksi2], [0, ksi2, ksi1]], dtype=torch.float64)


def b_vector(
    rawdata: torch.Tensor,
    angles: torch.Tensor,
    n: int,
    nDiff: int = 17,
    baseresolution: int = 256,
    Npad: int = 1000,
    beta: float = 10.0,
    readoutDownsampling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the vector b for gradient delay estimation.

    Args:
        rawdata (torch.Tensor): Raw k-space data.
        angles (torch.Tensor): Corresponding angles.
        n (int): Index.
        nDiff (int): Index difference.
        baseresolution (int): Base resolution.
        Npad (int): Padding size.
        beta (float): Scaling factor.
        readoutDownsampling (bool): Whether to downsample.

    Returns:
        tuple: Interpolated data, differentiation, index crossing, and b vector.
    """
    angle0 = angles[n + nDiff]
    angle1 = angles[n]

    data = rawdata[:, [n + nDiff, n], :].swapaxes(-1, -2)

    data_interpol = interpolation(
        data,
        baseresolution=baseresolution,
        Npad=Npad,
        beta=beta,
        readoutDownsampling=readoutDownsampling,
    )

    diff = differentiation(data_interpol)
    ind_cross = indexing(diff, Npad=Npad, beta=beta)

    b0 = ind_cross[0] * torch.tensor(
        [[torch.cos(angle0)], [torch.sin(angle0)]], dtype=torch.float64
    )
    b1 = ind_cross[1] * torch.tensor(
        [[torch.cos(angle1)], [torch.sin(angle1)]], dtype=torch.float64
    )

    b = b1 - b0

    return data_interpol, diff, ind_cross, b


def k_gd(
    k_cal: torch.Tensor,
    angles: torch.Tensor,
    sx: float = 0.5,
    sy: float = 0.5,
    sxy: float = 0.0,
    pixel: int = 1,
    spokes: int = 256,
    baseresolution: int = 256,
    correction: bool = False,
) -> torch.Tensor:
    """
    Generates distorted k-space trajectories due to gradient delays.

    Args:
        k_cal (torch.Tensor): Nominal k-space trajectory.
        angles (torch.Tensor): Corresponding angles.
        sx (float): Scaling factor in x-direction.
        sy (float): Scaling factor in y-direction.
        sxy (float): Scaling factor for skew.
        pixel (int): Pixel scaling factor.
        spokes (int): Number of spokes.
        baseresolution (int): Base resolution.
        correction (bool): Whether to apply correction.

    Returns:
        torch.Tensor: Distorted k-space trajectory.
    """
    distortion_array = torch.zeros((2, spokes), dtype=torch.float64)

    sx = pixel / baseresolution * sx
    sy = pixel / baseresolution * sy
    sxy = pixel / baseresolution * sxy

    for i in range(spokes):
        distortion_array[0, i] = sx * torch.cos(angles[i]) + sxy * torch.sin(angles[i])
        distortion_array[1, i] = sy * torch.sin(angles[i]) + sxy * torch.cos(angles[i])

    distortion = distortion_array.unsqueeze(-1)

    # Apply distortion to k-space
    k_distorted = k_cal - distortion if not correction else k_cal + distortion

    return k_distorted


def estimate_gradient_delay(
    rawdata: torch.Tensor,
    k_nom: torch.Tensor,
    shape: tuple,
    Npad: int = 100,
    beta: float = 5.0,
    nDiff: int = 17,
    nPairs: int = 10,
    readoutDownsampling: bool = True,
) -> torch.Tensor:
    """
    Estimates gradient delays using the RING method.

    Args:
        rawdata (torch.Tensor): Raw k-space data.
        k_nom (torch.Tensor): Nominal k-space trajectory.
        shape (tuple): Data shape.
        Npad (int): Padding size.
        beta (float): Scaling factor.
        nDiff (int): Index difference.
        nPairs (int): Number of pairs for estimation.
        readoutDownsampling (bool): Whether to downsample.

    Returns:
        torch.Tensor: Gradient delay correction.
    """
    spokes, baseresolution, nZ, nS, nTI, nTE = shape
    k_nom = k_nom.view(2, spokes, baseresolution, nZ, nS, nTI, nTE)
    angles = torch.atan2(k_nom[1, :, 0, ...], k_nom[0, :, 0, ...])

    A_total = torch.zeros((nTE, nTI, nS, nPairs, 2, 3), dtype=torch.float64)
    b_total = torch.zeros((nTE, nTI, nS, nPairs, 2, 1), dtype=torch.float64)

    for iTE in range(nTE):
        for iTI in range(nTI):
            for iS in range(nS):
                for iPair in range(nPairs):
                    A_total[iTE, iTI, iS, iPair, ...] = A_matrix(
                        angles[:, nZ // 2, iS, iTI, iTE], iPair, nDiff
                    )
                    b_total[iTE, iTI, iS, iPair, ...] = b_vector(
                        rawdata[:, :, :, nZ // 2, iS, iTI, iTE],
                        angles[:, nZ // 2, iS, iTI, iTE],
                        iPair,
                        nDiff=nDiff,
                        baseresolution=baseresolution,
                        Npad=Npad,
                        beta=beta,
                        readoutDownsampling=readoutDownsampling,
                    )[3]

    A_total = A_total.view(nTE, -1, 3)
    b_total = b_total.view(nTE, -1, 1)

    S_correction = torch.zeros((nTE, 3), dtype=torch.float64)
    for iTE in range(nTE):
        s = torch.linalg.lstsq(A_total[iTE, ...], b_total[iTE, ...])
        S_correction[iTE, :] = torch.squeeze(s.solution)

    return S_correction


def correct_kspace_trajectory(
    k_nom: torch.Tensor,
    shape: tuple,
    S_correction: torch.Tensor,
) -> torch.Tensor:
    """
    Applies the estimated gradient delay correction to the k-space trajectory.

    Args:
        k_nom (torch.Tensor): Nominal k-space trajectory.
        shape (tuple): Data shape.
        S_correction (torch.Tensor): Gradient delay correction.

    Returns:
        torch.Tensor: Corrected k-space trajectory.
    """
    spokes, baseresolution, nZ, nS, nTI, nTE = shape
    k_nom = k_nom.view(2, spokes, baseresolution, nZ, nS, nTI, nTE)
    angles = torch.atan2(k_nom[1, :, 0, ...], k_nom[0, :, 0, ...])

    k_gdc = torch.zeros_like(k_nom)

    for iTE in range(nTE):
        for iTI in range(nTI):
            for iS in range(nS):
                for iZ in range(nZ):
                    k_gdc[..., iZ, iS, iTI, iTE] = k_gd(
                        k_nom[..., iZ, iS, iTI, iTE],
                        angles[..., iZ, iS, iTI, iTE],
                        sx=S_correction[iTE, 0],
                        sy=S_correction[iTE, 1],
                        sxy=S_correction[iTE, 2],
                        spokes=spokes,
                        baseresolution=baseresolution,
                        correction=True,
                    )

    return k_gdc
