import torch


def radial_trajectory_3d(
    t1: int,
    t2: int,
    z: int,
    nT1: int,
    nT2: int,
    nZ: int,
    baseresolution: int,
    spokes: int,
    phi0: float = -torch.pi / 2,
    readoutDownsampling: bool = True,
    version: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a 3D radial trajectory for MRI based on the paper:
    Zhou, Z., Han, F., Yan, L., Wang, D.J.J., Hu, P., 2017.
    Golden-ratio rotated stack-of-stars acquisition for improved volumetric MRI.
    Magn. Reson. Med. 78, 2290–2298. https://doi.org/10.1002/mrm.26625

    Args:
        t1 (int): T1 index.
        t2 (int): T2 index.
        z (int): Slice index.
        nT1 (int): Number of T1 indices.
        nT2 (int): Number of T2 indices.
        nZ (int): Number of slices.
        baseresolution (int): Base resolution.
        spokes (int): Number of radial spokes.
        phi0 (float): Initial rotation angle.
        readoutDownsampling (bool): Whether to downsample readout.
        version (int): Version of trajectory computation.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Cartesian k-space coordinates (Kx, Ky).
    """
    # Golden angle
    GA = 2 * torch.pi / (1 + torch.sqrt(torch.tensor(5)))

    if version == 1:
        # Continuous golden angle ordering
        phi = (
            t2 + t1 * nT2 + torch.arange(spokes, dtype=torch.float64) * nT1 * nT2
        ) * GA
        phi += z * spokes * GA  # Rotated stack-of-stars (RSOS) trajectory

    elif version == 2:
        phi = (
            t2 / (nT1 * nT2 * nZ)
            + t1 / (nT1 * nZ)
            + z / nZ
            + torch.arange(spokes, dtype=torch.float64)
        ) * GA

    # Rotate by phi0 (so that head is up)
    phi += phi0

    # Rotate every echo by PI to minimize gradients
    phi += t2 * torch.pi

    # Generate rho values for the radial distance
    rho = torch.linspace(
        -0.5, 0.5, baseresolution + int(readoutDownsampling), dtype=torch.float64
    )
    if readoutDownsampling:
        rho = rho[:-1]

    # Generate K-space trajectory in polar coordinates
    K = rho[None, :] * torch.exp(1j * phi[:, None])

    # Convert polar to Cartesian coordinates
    Kx, Ky = torch.real(K), torch.imag(K)

    return Kx, Ky
