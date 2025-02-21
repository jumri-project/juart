import torch


def geometric_coil_compression_matrices(
    data: torch.Tensor,
    ncoils: int,
) -> torch.Tensor:
    """
    Compute the coil compression matrices for geometric coil compression.

    Args:
        data (torch.Tensor): Input data with shape (coils, ... , slices).
        ncoils (int): Number of coils to retain after compression.

    Returns:
        torch.Tensor: Compression matrices.
    """
    nC = data.shape[0]  # Number of coils
    nS = data.shape[4]  # Number of slices

    # Reshape data for SVD
    data = data.transpose(4, -1).reshape(nC, -1, nS)

    # Initialize the compression matrix
    A = torch.zeros((nC, nC, nS), dtype=data.dtype)

    # Perform SVD and compute the compression matrix
    for iS in range(nS):
        U, _, _ = torch.linalg.svd(data[:, :, iS], full_matrices=False)
        A[:, :, iS] = torch.conj(U.T)

    # Retain only the top `ncoils` components
    A = A[:ncoils, :, :]

    # Alignment step (Paper Zhang, MRM, 2012)
    for iS in range(nS - 1):
        Cx = torch.matmul(A[..., iS + 1], torch.conj(A[..., iS].T))
        Uc, _, Vhc = torch.linalg.svd(Cx)
        Px = torch.matmul(torch.conj(Vhc.T), torch.conj(Uc.T))
        A[..., iS + 1] = torch.matmul(Px, A[..., iS + 1])

    return A


def apply_geometric_coil_compression(
    data: torch.Tensor,
    A: torch.Tensor,
    ncoils: int,
) -> torch.Tensor:
    """
    Apply the coil compression matrices to the input data.

    Args:
        data (torch.Tensor): Input data with shape (coils, ..., slices).
        A (torch.Tensor): Compression matrices.
        ncoils (int): Number of coils to retain after compression.

    Returns:
        torch.Tensor: Compressed data.
    """
    nC = data.shape[0]  # Number of coils
    nS = data.shape[4]  # Number of slices

    # Reshape data for matrix multiplication
    data = data.transpose(4, -1)
    shape = data.shape
    data = data.reshape(nC, -1, nS)

    # Apply the compression matrix
    for iS in range(nS):
        data[:ncoils, ..., iS] = torch.matmul(A[..., iS], data[..., iS])

    # Reshape data back to its original form
    data = data.reshape(shape)
    data = data[:ncoils, ...]

    # Swap axes back to the original order
    data = data.transpose(4, -1)

    return data
