from typing import Optional, Tuple, Union

import torch
from torch import jit


@jit.script
def singular_value_soft_thresholding_kernel(
    x: torch.Tensor,
    lamda: float,
    chunk_size: int = 16384,
    epsilon: float = 1e-9,
    driver: Union[str, None] = "gesvda",
) -> torch.Tensor:
    """
    Singular value soft thresholding kernel using PyTorch.

    This function applies singular value decomposition (SVD) on the input tensor `x`,
    performs soft thresholding on the singular values using the given threshold `lamda`,
    and then reconstructs the tensor with the thresholded singular values.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor to which singular value soft thresholding is applied.
    lamda : float
        Threshold parameter for soft thresholding of singular values.
    """

    if x.device == "cpu":
        driver = None

    shape = x.shape
    x = x.reshape(-1, x.shape[-2], x.shape[-1])

    x = x + torch.randn(x.shape, dtype=x.dtype, device=x.device) * epsilon

    num_chunks = (x.shape[0] + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        x_chunk = x[i * chunk_size : (i + 1) * chunk_size, :, :]

        U, S, V = torch.linalg.svd(x_chunk, full_matrices=False, driver=driver)
        S_thresh = torch.clamp(S - lamda, min=0)
        x_chunk = torch.matmul(U * S_thresh.unsqueeze(-2), V)

        x[i * chunk_size : (i + 1) * chunk_size, :, :] = x_chunk

    x = x.reshape(shape)

    return x


def singular_value_soft_thresholding(
    x: torch.Tensor,
    lamda: float,
    transpose: Optional[Tuple[int, ...]] = None,
    reshape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Singular value soft thresholding function with optional transpose and reshape.

    This function applies singular value soft thresholding to the input tensor `x`,
    with optional transposing and reshaping steps. It handles any non-finite values
    (such as NaNs or infinities) in the input tensor by setting them to zero.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor to be thresholded.
    lamda : float
        Threshold parameter for singular value soft thresholding.
    transpose : tuple, optional
        Transpose dimensions, if required.
    reshape : tuple, optional
        Reshape dimensions, if required.

    Returns:
    -------
    torch.Tensor
        The tensor after applying singular value soft thresholding.
    """
    if transpose is not None:
        x = x.permute(transpose)

    if reshape is not None:
        shape = x.shape
        x = x.reshape(reshape)

    # Remove non-finite values (inf or NaN)
    x[~torch.isfinite(x)] = 0

    try:
        x = singular_value_soft_thresholding_kernel(x, lamda)
    except Exception as e:
        print("singular_value_soft_thresholding failed:", str(e))

    if reshape is not None:
        x = x.reshape(shape)

    if transpose is not None:
        x = x.permute(tuple(torch.argsort(torch.tensor(transpose))))

    return x


class SingularValueSoftThresholdingOperator:
    """
    Singular Value Soft Thresholding Operator using PyTorch.

    This operator provides a method for applying singular value soft thresholding on a
    tensor. It can optionally reshape and transpose the input tensor before thresholding
    and revert the transformations afterward.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        lamda: float,
        transpose: Optional[Tuple[int, ...]] = None,
        reshape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Initialize the SingularValueSoftThresholdingOperator.

        Parameters:
        ----------
        shape : tuple
            Shape of the input tensor.
        lamda : float
            Threshold parameter for singular value soft thresholding.
        transpose : tuple, optional
            Transpose dimensions, if required.
        reshape : tuple, optional
            Reshape dimensions, if required.
        """
        self.shape = shape
        self.lamda = lamda
        self.transpose = transpose
        self.reshape = reshape

        self.dtype = torch.float32
        self.internal_dtype = torch.complex64

    def solve(
        self,
        x: torch.Tensor,
        rho: float,
        *args,
    ) -> torch.Tensor:
        """
        Solve the singular value soft thresholding problem.

        This function applies singular value soft thresholding on the input tensor `x`
        using the provided regularization parameter `rho`.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to which singular value soft thresholding is applied.
        rho : float
            Regularization parameter.

        Returns:
        -------
        torch.Tensor
            The updated tensor after applying singular value soft thresholding.
        """
        x = x.view(self.internal_dtype).reshape(self.shape)
        x = singular_value_soft_thresholding(
            x,
            self.lamda / rho,
            transpose=self.transpose,
            reshape=self.reshape,
        )
        x = x.ravel().view(self.dtype)

        return x
