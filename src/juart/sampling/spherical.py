import torch


def spherical_trajectory_3d(
    baseresolution,
    projections,
    readoutDownsampling: bool = True,
):

    phi_1 = 0.4656
    phi_2 = 0.6823

    m = torch.arange(projections, dtype=torch.float32)

    alpha = torch.arccos(torch.remainder(m * phi_1, 1.0))
    beta = 2 * torch.pi * torch.remainder(m * phi_2, 1.0)

    # Generate rho values for the radial distance
    rho = torch.linspace(
        -0.5, 0.5, baseresolution + int(readoutDownsampling), dtype=torch.float32
    )
    if readoutDownsampling:
        rho = rho[:-1]

    x = rho[:, None] * torch.sin(alpha[None, :]) * torch.cos(beta[None, :])
    y = rho[:, None] * torch.sin(alpha[None, :]) * torch.sin(beta[None, :])
    z = rho[:, None] * torch.cos(alpha[None, :])

    k = torch.stack((x, y, z))

    return k
